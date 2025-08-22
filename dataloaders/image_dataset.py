#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import logging
from argparse import Namespace
import json
import exifread
from dataloaders.read_write_model import read_model, qvec2rotmat
from utils import get_image_names


class ImageDataset:
    """
    The main dataset class for loading images from disk in a multithreaded manner.
    It also supports loading masks and COLMAP poses if available.
    The next image can be fetched using the `getnext` method.
    """
    def __init__(self, args: Namespace):
        self.images_dir = os.path.join(args.source_path, args.images_dir)
        self.image_name_list = get_image_names(self.images_dir)
        self.image_name_list.sort()
        self.image_name_list = self.image_name_list[args.start_at :]
        self.image_paths = [
            os.path.join(self.images_dir, image_name)
            for image_name in self.image_name_list
        ]
        if len(self.image_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.images_dir}")

        self.mask_dir = (
            os.path.join(args.source_path, args.masks_dir) if args.masks_dir else None
        )
        if self.mask_dir:
            self.mask_paths = [
                os.path.join(self.mask_dir, os.path.splitext(image_name)[0] + ".png")
                for image_name in self.image_name_list
            ]
            assert all(os.path.exists(mask_path) for mask_path in self.mask_paths), (
                "Not all masks exist."
            )

        self.downsampling = args.downsampling
        self.num_threads = min(args.num_loader_threads, len(self.image_paths))
        self.current_index = 0
        self.preload_queue = Queue(maxsize=self.num_threads)
        self.executor = ThreadPoolExecutor(max_workers=self.num_threads)

        self.infos = {
            name: {
                "is_test": (args.test_hold > 0) and (i % args.test_hold == 0),
                "name": name,
            }
            for i, name in enumerate(self.image_name_list)
        }

        first_image = self._load_image(self.image_paths[0])
        self.width, self.height = first_image.shape[2], first_image.shape[1]
        res = self.width * self.height
        max_res = 1_500_000  # 1.5 Mpx
        if self.downsampling <= 0.0 and res > max_res:
            logging.warning(
                "Large images, downsampling to 1.5 Mpx. "
                "If this is not desired, please use --downsampling=1"
            )
            self.downsampling = (res / max_res) ** 0.5
            first_image = self._load_image(self.image_paths[0])
            self.width, self.height = first_image.shape[2], first_image.shape[1]


        if args.info_type == "json": 
            info_path = os.path.join(args.source_path, "info")
            self.load_json_data(info_path)
        elif args.info_type == "exif":
            self.load_exif_data(os.path.join(args.source_path, "images"))
        elif args.info_type == "colmap":
            self.load_colmap_data(os.path.join(args.source_path, "sparse/0"))
        
        if args.gt_type == "colmap":
            self.load_colmap_data(os.path.join(args.source_path, "sparse/0"),  pose_key="Rt")
        elif args.gt_type == "exif":
            self.load_exif_data(os.path.join(args.source_path, "images"), pose_key="Rt")
            

        # Check that all images have poses
        has_all_poses = all(
            "precomputed_pose" in self.infos[image_name] for image_name in self.image_name_list
        )
        has_all_gt_poses = all(
            "Rt" in self.infos[image_name] for image_name in self.image_name_list
        )
        if args.use_precomputed_poses:
            assert has_all_poses, (
                "Precomputed poses are required but not all images have poses."
            )
            self.align_precomputed_poses()

        if args.eval_poses and not has_all_gt_poses:
            logging.warning(
                " Not all images have COLMAP poses, pose evaluation will be skipped."
            )


        self.train_indices = []
        self.holdout_indices = []
        self.holdout_name_list = []
        
        full_image_name_list = sorted(get_image_names(self.images_dir))

        holdout_names = {full_image_name_list[i] for i in args.holdout_frames if i < len(full_image_name_list)}
        
        for i, name in enumerate(self.image_name_list):
            if name in holdout_names:
                self.holdout_indices.append(i)
                self.holdout_name_list.append(name)
            else:
                self.train_indices.append(i)

        self.train_image_paths = [self.image_paths[i] for i in self.train_indices]
        self.holdout_image_paths = [self.image_paths[i] for i in self.holdout_indices]

        self.start_preloading()

    def get_holdout_item(self, holdout_idx):
        """Gets a specific item from the holdout set."""
        image_path = self.holdout_image_paths[holdout_idx]
        image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
        info = self.infos[os.path.basename(image_path)]
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].cpu()
            image = image[:3]
        return image.cuda(), info

    def __len__(self):
        return len(self.train_image_paths)

    @torch.no_grad()
    def __getitem__(self, index):
        original_idx = self.train_indices[index]
        image_path = self.image_paths[original_idx] # Use original full path list
        image = self._load_image(image_path, cv2.IMREAD_UNCHANGED)
        info = self.infos[os.path.basename(image_path)]
        if image.shape[0] == 4:
            info["mask"] = image[-1][None].cpu()
            image = image[:3]
        if self.mask_dir:
            mask = self._load_image(self.mask_paths[original_idx])
            info["mask"] = mask[0][None]
        return image.cuda(), info

    def _load_image(self, image_path, mode=cv2.IMREAD_COLOR):
        image = cv2.imread(image_path, mode)
        if image is None:
            raise FileNotFoundError(f"Image at {image_path} could not be loaded.")
        if self.downsampling > 0.0 and self.downsampling != 1.0:
            image = cv2.resize(
                image,
                (0, 0),
                fx=1 / self.downsampling,
                fy=1 / self.downsampling,
                interpolation=cv2.INTER_AREA,
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if image.shape[-1] == 4 else cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image

    def _submit(self):
        if self.current_index < len(self):
            self.preload_queue.put(
                self.executor.submit(self.__getitem__, self.current_index)
            )

    def start_preloading(self):
        """Start threads to preload images."""
        for self.current_index in range(self.num_threads):
            self._submit()

    def getnext(self):
        """Get the next item from the dataset and start preloading the next one."""
        item = self.preload_queue.get().result()
        self.current_index += 1
        self._submit()
        return item

    def get_image_size(self):
        return self.height, self.width

    def load_exif_data(self, image_dir_path, pose_key="precomputed_pose"):
        for image_name in self.image_name_list:
            image_path = os.path.join(image_dir_path, image_name)
        
            with open(image_path, 'rb') as f:
                tags = exifread.process_file(f)

            comment_tag = tags.get('EXIF UserComment')

            if not comment_tag:
                print(f"Warning: No 'EXIF UserComment' tag found in {image_name}")
                continue

            comment_str = comment_tag.values
            
            json_start_index = comment_str.find('{')
            if json_start_index == -1:
                print(f"Warning: Could not find JSON start in UserComment for {image_name}")
                continue
            
            json_str = comment_str[json_start_index:]
            
            pose_data = json.loads(json_str)
            pos = pose_data['position']
            rot = pose_data['rotation']

            c2w = torch.eye(4, dtype=torch.float32)
            c2w[:3, :3] = torch.from_numpy(qvec2rotmat([rot['w'], -rot['x'], rot['y'], -rot['z']]))
            c2w[:3, 3] = torch.from_numpy(np.array([pos['x'], -pos['y'], pos['z']]))
    
            w2c = torch.inverse(c2w)
            
            self.infos[image_name][pose_key] = w2c.cuda()

    def load_json_data(self, info_dir_path):
        """Load camera intrinsics and extrinsics from custom JSON files."""
        T_world = torch.diag(torch.tensor([1., 1., -1., 1.])).float()

        T_cam = torch.diag(torch.tensor([1., -1., 1., 1.])).float()

        for image_name in self.image_name_list:
            base_name = os.path.splitext(image_name)[0]
            json_name = base_name.replace("Image_", "Metadata_") + ".json"
            json_path = os.path.join(info_dir_path, json_name)

            if not os.path.exists(json_path):
                print(f"Warning: Could not find JSON info file for {image_name}")
                continue

            with open(json_path, 'r') as f:
                data = json.load(f)

            intrinsics = data["intrinsics"]
            fx = intrinsics["fx"]
            fy = intrinsics["fy"]
            # The framework uses a single focal length, so we average them
            focal = (fx + fy) / 2.0
            
            matrix_c2w_opengl = torch.tensor(data["cameraToWorldMatrix"]).reshape(4, 4).T #transpose to convert to row-major

            temp_matrix_c2w_opencv =  matrix_c2w_opengl @ T_cam 

            matrix_c2w_opencv = T_world @ temp_matrix_c2w_opencv        
            
            matrix_w2c = torch.inverse(matrix_c2w_opencv)

            self.infos[image_name]["precomputed_pose"] = matrix_w2c.cuda()
    
    def load_colmap_data(self, colmap_folder_path, pose_key="precomputed_pose"):
        """Load COLMAP camera intrinsics and extrinsics. Stores them in self.infos."""
        try:
            cameras, images, _ = read_model(colmap_folder_path)
        except Exception as e:
            logging.warning(
                f" Failed to read COLMAP files in {colmap_folder_path}: {e}"
            )
            return
        if len(cameras) != 1:
            logging.warning(" Only supports one camera")
        model = list(cameras.values())[0].model
        if model != "PINHOLE" and model != "SIMPLE_PINHOLE":
            logging.warning(" Unexpected camera model: " + model)

        for image_id, image in images.items():
            camera = cameras[image.camera_id]

            # Intrinsics and projection matrix
            focal_x = camera.params[0]
            focal_y = camera.params[1] if camera.model == "PINHOLE" else focal_x
            focal = (focal_x + focal_y) / 2
            focal = focal * self.width / camera.width

            # Pose
            Rt = np.eye(4, dtype=np.float32)
            Rt[:3, :3] = qvec2rotmat(image.qvec)
            Rt[:3, 3] = image.tvec

            name = os.path.basename(image.name)
            if image.name in self.infos:
                self.infos[name][pose_key] = torch.tensor(Rt, device="cuda")

    def align_precomputed_poses(self, pose_key="precomputed_pose"):
        """Scale and set first Rt as identity"""
        centres = []
        for idx in range(6):
            centres.append(self.infos[self.image_name_list[idx]][pose_key].inverse()[:3, 3])
        centres = torch.stack(centres)
        rel_ts = centres[:-1] - centres[1:]

        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        inv_first_Rt = self.infos[self.image_name_list[0]][pose_key].inverse()
        for info in self.infos.values():
            info[pose_key] = info[pose_key] @ inv_first_Rt
            info[pose_key][:3, 3] *= scale