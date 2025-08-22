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

import torch
import math

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from utils import fov2focal, depth2points, sixD2mtx, mtx2sixD, pts2px
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType
import cv2

class PoseInitializer():
    """Fast pose initializer using MiniBA and the previous frames."""
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher
        self.alignment_transform = None

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers

        # Initialize the focal length
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        self.miniba_bootstrap = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  not args.fix_focal, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniba_rebooting = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=True, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        unused_kpts_mask = torch.ones((n_cams, desc_kpts_list[0].kpts.shape[0]), device='cuda', dtype=torch.bool)
        for k in range(n_primary_cam):
            idx_occurrences = torch.zeros(self.num_kpts, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            selected_indices = torch.multinomial(idx_occurrences.float(), npts_per_primary_cam, replacement=False)

            selected_mask = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    def triangulate_tracks(self, uvs, infos):
        num_ba_points = uvs.shape[0]
        point_cloud_ba = torch.zeros(num_ba_points, 3, device="cuda")
        point_cloud_validity = torch.zeros(num_ba_points, dtype=torch.bool, device="cuda")

        for i in range(num_ba_points):
            track = uvs[i]
            valid_view_indices = torch.where(track[:, 0] != -1)[0]

            if len(valid_view_indices) < 2:
                continue

            best_point = None
            max_angle = -1.0

            for j in range(len(valid_view_indices)):
                for k in range(j + 1, len(valid_view_indices)):
                    cam_idx1 = valid_view_indices[j]
                    cam_idx2 = valid_view_indices[k]

                    uv1 = track[cam_idx1]
                    uv2 = track[cam_idx2]
                    
                    Rt1 = infos[cam_idx1]["precomputed_pose"]
                    Rt2 = infos[cam_idx2]["precomputed_pose"]
                    
                    center1 = -Rt1[:3, :3].T @ Rt1[:3, 3]
                    center2 = -Rt2[:3, :3].T @ Rt2[:3, 3]
                    
                    focal = self.f_init
                    ray1 = torch.nn.functional.normalize(torch.tensor([(uv1[0] - self.width/2)/focal, (uv1[1] - self.height/2)/focal, 1.0], device="cuda"), dim=0)
                    ray2 = torch.nn.functional.normalize(torch.tensor([(uv2[0] - self.width/2)/focal, (uv2[1] - self.height/2)/focal, 1.0], device="cuda"), dim=0)
                    
                    ray1_world = (Rt1[:3, :3].T @ ray1)
                    ray2_world = (Rt2[:3, :3].T @ ray2)
                    
                    angle = torch.acos(torch.dot(ray1_world, ray2_world)) * (180.0 / math.pi)

                    if angle > max_angle:
                        max_angle = angle
                        
                        K = torch.tensor([[focal, 0, self.width/2.0], [0, focal, self.height/2.0], [0, 0, 1]], device="cuda")
                        proj1 = K @ Rt1[:3, :]
                        proj2 = K @ Rt2[:3, :]
                        
                        points_4d = cv2.triangulatePoints(proj1.cpu().numpy(), proj2.cpu().numpy(), 
                                                          uv1.cpu().numpy().reshape(2,1), uv2.cpu().numpy().reshape(2,1))
                        point_3d = torch.from_numpy(points_4d[:3] / points_4d[3]).T.cuda().float()
                        best_point = point_3d

            if max_angle > math.radians(10.0) and best_point is not None:
                point_cloud_ba[i] = best_point
                point_cloud_validity[i] = True
        
        print(f"Successfully triangulated {point_cloud_validity.sum()} / {num_ba_points} points.")
        return point_cloud_ba, point_cloud_validity

    @torch.no_grad()
    def initialize_bootstrap_refine(self, desc_kpts_list: list[DescribedKeypoints], bootstrap_infos=None):
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)        

        uvs, xyz_indices = self.build_problem(desc_kpts_list, self.num_pts_miniba_bootstrap, n_cams, n_cams, 2, list(range(n_cams)))

        point_cloud_ba, valid_mask = self.triangulate_tracks(uvs, bootstrap_infos)

        uvs[~valid_mask, :, :] = -1

        Rs6D_init = torch.stack([mtx2sixD(info["precomputed_pose"][:3, :3]) for info in bootstrap_infos], dim=0).to("cuda")
        ts_init = torch.stack([info["precomputed_pose"][:3, 3] for info in bootstrap_infos], dim=0).to("cuda")
        f_init = torch.tensor([self.f_init], device="cuda")

        Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, point_cloud_ba, self.centre, uvs.view(-1))

        final_residual = (r * mask).abs().sum()/mask.sum()
        print(f"Final residual: {final_residual.item()}")

        self.f = f

        Rts_refined = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts_refined[:, :3, :3] = sixD2mtx(Rs6D)
        Rts_refined[:, :3, 3] = ts

        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()

        inv_first_Rt = torch.linalg.inv(Rts_refined[0])

        self.alignment_transform = {"inv_first_Rt": inv_first_Rt, "scale": scale}
        
        final_Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        for i in range(n_cams):
           aligned_pose = Rts_refined[i] @ inv_first_Rt
           aligned_pose[:3, 3] *= scale
           final_Rts[i] = aligned_pose

        return final_Rts, f, final_residual


    @torch.no_grad()
    def initialize_bootstrap(self, desc_kpts_list: list[DescribedKeypoints], bootstrap_infos=None, rebooting=False):
        """
        Estimate focal and initialize the poses of the frames corresponding to desc_kpts_list. 
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
        
        ## Build the problem by organizing matches
        uvs, xyz_indices = self.build_problem(desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams)))

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        f_init = (torch.tensor([self.f_init], device="cuda"))
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        ## Run miniBA, estimating 3D points, camera focal and poses
        if rebooting:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(Rs6D_init, ts_init, self.f, xyz_init, self.centre, uvs.view(-1))
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1))
        final_residual = (r * mask).abs().sum()/mask.sum()

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0)

        ## Scale to 0.1 average translation
        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        ts *= scale
        xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts

        return Rts, f, final_residual

    @torch.no_grad()
    def initialize_incremental_refine(self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints, index: int, is_test: bool, image, incr_info=None):
        Rt = incr_info["precomputed_pose"]
        aligned_Rt = Rt @ self.alignment_transform["inv_first_Rt"]
        aligned_Rt[:3, 3] *= self.alignment_transform["scale"] 
        Rt = aligned_Rt
        
        final_xyz_matches = []
        final_uvs_matches = []
        

        print(f"Attempting to match against {len(keyframes)} previous keyframes.")
        for kf in keyframes:
            matches = self.matcher(curr_desc_kpts, kf.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=index, kID_other=kf.index)
            
            if len(matches.idx) == 0:
                continue # No visual overlap found with this keyframe
            
            old_kf_indices_in_match = matches.idx_other
            mask_has_3d = kf.desc_kpts.has_pt3d[old_kf_indices_in_match]
            
            if mask_has_3d.sum() == 0:
                continue # Matches were found, but none of them have 3D points

            xyz_world = kf.desc_kpts.pts3d[old_kf_indices_in_match[mask_has_3d]]
            uvs_new = matches.kpts[mask_has_3d]

            # h, w = image.shape[1], image.shape[2]
            # image_dims = (w, h)
            # visualize_3d_reprojection(Rt, kf, xyz_world, uvs_new, self.f, self.centre, image_dims)
            # print("Exiting after first visualization for debugging.")
            # exit() # Stop the program here so we can analyze the plot 

            R_w2c, t_w2c = Rt[:3, :3], Rt[:3, 3]
            xyz_cam = (R_w2c @ xyz_world.T).T + t_w2c
            
            # Filter points behind the camera
            # valid_depth_mask = xyz_cam[:, 2] > 1e-3
            # if valid_depth_mask.sum() == 0:
            #     continue
            valid_depth_mask = torch.ones(xyz_cam.shape[0], device="cuda", dtype=torch.bool)

            uvs_projected = pts2px(xyz_cam[valid_depth_mask], self.f, self.centre)
            
            error = torch.linalg.norm(uvs_new[valid_depth_mask] - uvs_projected, dim=-1)
            
            inlier_mask = error < (self.max_pnp_error * 5)
                
            num_inliers = inlier_mask.sum().item()

            if num_inliers == 0:
                print(f"  - KF {kf.index}: Found {len(matches.idx)} 2D matches, but all {mask_has_3d.sum()} 3D links failed reprojection check.")
                continue
                
            print(f"  - KF {kf.index}: Found {num_inliers} consistent 2D-3D correspondences.")
            
            # 5. Store the successful correspondences
            final_uvs_matches.append(uvs_new[valid_depth_mask][inlier_mask])
            final_xyz_matches.append(xyz_world[valid_depth_mask][inlier_mask])
            

        if not final_xyz_matches:
            print("Too few inliers for pose initialization after matching.")
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None
            
        xyz = torch.cat(final_xyz_matches, dim=0)
        uvs = torch.cat(final_uvs_matches, dim=0)

        print(f"Found a total of {len(xyz)} 2D-3D matches for refinement.")
        
        # Subsample points for MiniBA (same logic as before)
        if len(xyz) >= self.num_pts_miniba_incr:
            perm = torch.randperm(len(xyz))
            selected_indices = perm[:self.num_pts_miniba_incr]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        else:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)
        
        # 5. Run MiniBA for refinement (same logic as before)
        from utils import mtx2sixD
        Rs6D, ts = mtx2sixD(Rt[:3, :2])[None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        
        final_Rt = torch.eye(4, device="cuda")
        final_Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        final_Rt[:3, 3] = ts[0]
        
        # A final sanity check on the number of inliers after BA
        if mask.sum() < self.min_num_inliers:
            print("Too few inliers after final BA refinement.")
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None

        return final_Rt

    @torch.no_grad()
    def initialize_incremental(self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints, index: int, is_test: bool, curr_img, incr_info=None):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """
        
        # Match the current frame with previous keyframes
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        for keyframe in keyframes:
            matches = self.matcher(curr_desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=index, kID_other=keyframe.index)

            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])

        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        Rt, inliers = self.PnPRANSAC(uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs)

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]

        # Subsample the points if there are too many
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

        # Run the initialization
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]

        # Check if we have sufficiently many inliers
        if is_test or mask.sum() > self.min_num_inliers:
            # Return the pose of the current frame
            return Rt
        else:
            print("Too few inliers for pose initialization")
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None