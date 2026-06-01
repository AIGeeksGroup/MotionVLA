
import torch
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

# Rotation helpers
def rot6d_to_mat3x3(rot6d):
    rot6d = rot6d.reshape(-1, 3, 2)
    a1 = rot6d[:, :, 0]
    a2 = rot6d[:, :, 1]
    b1 = torch.nn.functional.normalize(a1, dim=-1)
    b2 = torch.nn.functional.normalize(a2 - torch.sum(b1 * a2, dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)

def mat3x3_to_axis_angle(rot_mat):
    device = rot_mat.device
    rot_mat_np = rot_mat.detach().cpu().numpy()
    rot_mat_np = rot_mat_np.reshape(-1, 3, 3)
    r = R.from_matrix(rot_mat_np)
    aa = r.as_rotvec()
    return torch.from_numpy(aa).to(device).float().reshape(rot_mat.shape[:-2] + (3,))

def vimogen_to_humanml3d_dual_stream(v_276_path):
    """
    Convert ViMoGen 276-dim data to HumanML3D 263-dim + Phys Residual.
    Uses STANDARD HumanML3D logic (Root-Relative).
    """
    print(f"Loading ViMoGen data from {v_276_path}")
    data = torch.load(v_276_path, map_location='cpu')
    
    dart_motion = None
    if isinstance(data, dict):
        if 'motion' in data: dart_motion = data['motion']
        # Fallback
        if dart_motion is None:
            for k, v in data.items():
                if isinstance(v, (torch.Tensor, np.ndarray)) and v.shape[-1] == 276:
                    dart_motion = v
                    break
    elif isinstance(data, (torch.Tensor, np.ndarray)):
        dart_motion = data
        
    if isinstance(dart_motion, np.ndarray): dart_motion = torch.from_numpy(dart_motion)
    dart_motion = dart_motion.float()
    
    T = dart_motion.shape[0]
    
    # 1. Parse DART layout
    # 0:126   - body_pose_6d (21*6)
    # 126:192 - joints (22*3) -> Index 0 is Root (Pelvis)
    # 192:258 - joints_vel (22*3)
    # 258:264 - root_orient_6d (6)
    # 264:270 - root_vel_6d (6)
    # 270:273 - root_trans (3)
    # 273:276 - root_trans_vel (3)
    
    body_pose_6d = dart_motion[:, 0:126]
    joints = dart_motion[:, 126:192].reshape(-1, 22, 3)
    joints_vel = dart_motion[:, 192:258].reshape(-1, 22, 3)
    root_orient_6d = dart_motion[:, 258:264]
    root_vel_6d = dart_motion[:, 264:270]
    root_trans = dart_motion[:, 270:273]
    root_trans_vel = dart_motion[:, 273:276]
    
    # 2. Compute Root Heading (Y-rotation)
    # Standard T2M logic: Heading is the rotation around Y axis.
    # We extract it from root_orient.
    # Assume Z is forward, Y is up.
    
    root_rot_mat = rot6d_to_mat3x3(root_orient_6d) # (T, 3, 3)
    
    # Global forward = R * (0, 0, 1) = R[:, 2] (3rd column)
    forward = root_rot_mat[:, :, 2] # (T, 3)
    # Heading angle = atan2(x, z)
    heading = torch.atan2(forward[:, 0], forward[:, 2]) # (T,)
    
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    
    def rotate_y_inv(vecs):
        # Rotate by -heading around Y
        if vecs.dim() < 2: vecs = vecs.unsqueeze(0)
        x = vecs[..., 0]
        y = vecs[..., 1]
        z = vecs[..., 2]
        
        dims_to_add = x.dim() - 1
        shape_expand = [-1] + [1] * dims_to_add
        c = cos_h.view(*shape_expand)
        s = sin_h.view(*shape_expand)
        
        x_new = x * c + z * s
        y_new = y
        z_new = -x * s + z * c
        return torch.stack([x_new, y_new, z_new], dim=-1)

    # 3. Compute Features
    
    # --- r_velocity (1) ---
    # Angular velocity around Y.
    # Use root_vel_6d (relative rotation).
    # Convert to axis-angle and take Y component.
    root_vel_mat = rot6d_to_mat3x3(root_vel_6d)
    root_vel_aa = mat3x3_to_axis_angle(root_vel_mat) # (T, 3)
    r_velocity = root_vel_aa[:, 1:2] # (T, 1)

    # --- l_velocity (2) ---
    # Linear velocity (XZ) in root frame.
    root_trans_vel_local = rotate_y_inv(root_trans_vel) # (T, 3)
    l_velocity = root_trans_vel_local[:, [0, 2]] # (T, 2) (X, Z)
    
    # --- root_y (1) ---
    root_y = root_trans[:, 1:2] # (T, 1)
    
    # --- ric_data (63) ---
    # Joints positions relative to root, rotated.
    # Exclude root joint (index 0).
    root_pos = joints[:, 0:1, :] # (T, 1, 3)
    joints_rel = joints - root_pos # (T, 22, 3)
    joints_rel_local = rotate_y_inv(joints_rel)
    ric_data = joints_rel_local[:, 1:, :].reshape(T, -1) # (T, 63)
    
    # --- rot_data (126) ---
    # Local rotations (relative to parent).
    # body_pose_6d is already local.
    rot_data = body_pose_6d # (T, 126)
    
    # --- vel_data (66) ---
    # Joint velocities (global velocities rotated to local frame).
    joints_vel_local = rotate_y_inv(joints_vel)
    vel_data = joints_vel_local.reshape(T, -1) # (T, 66)
    
    # --- foot_contact (4) ---
    # Indices: L_Ankle(7), R_Ankle(8), L_Foot(10), R_Foot(11) in SMPL 24 joints?
    # DART 22 joints indices might differ.
    # Usually: 0-Pelvis, ..., 7-L_Ankle, 8-R_Ankle, 10-L_Foot, 11-R_Foot.
    # Let's check magnitude of global velocity.
    fid_l = [7, 10]
    fid_r = [8, 11]
    
    vel_l = (joints_vel[:, fid_l, :] ** 2).sum(dim=-1)
    vel_r = (joints_vel[:, fid_r, :] ** 2).sum(dim=-1)
    
    contact_l = (vel_l < 0.00002).float()
    contact_r = (vel_r < 0.00002).float()
    
    foot_contact = torch.cat([contact_r, contact_l], dim=1) # (T, 4)
    
    # Concatenate 263
    base_263 = torch.cat([
        r_velocity,    # 1
        l_velocity,    # 2
        root_y,        # 1
        ric_data,      # 63
        rot_data,      # 126
        vel_data,      # 66
        foot_contact   # 4
    ], dim=1) # Total 263
    
    # Phys Residual (13 dims)
    # We need to store global root info to reconstruct exactly 276.
    # 276 = 263 + X? No, 276 and 263 represent same motion but different frames.
    # But if we want PERFECT reconstruction of global pose, we need initial state or full global trajectory.
    # 263 is root-relative (invariant).
    # To get back to 276 (Global), we need:
    # 1. Root Global Orient (6)
    # 2. Root Global Pos (3)
    # 3. Root Rot Vel (6) -> approximated by r_velocity (1), but maybe we want exact 6D?
    # 4. Root Lin Vel (3) -> approximated by l_velocity (2) + root_y change?
    #
    # Let's store the exact global root parameters as "residual".
    # We need to reconstruct:
    # - root_orient_6d (6)
    # - root_trans (3)
    # - root_vel_6d (6)
    # - root_trans_vel (3)
    # Total 18 dims of root state.
    # 263 has r_vel(1), l_vel(2), root_y(1). Total 4 dims of root info.
    # We are missing absolute XZ position and absolute Heading.
    #
    # Phys Residual (13 dims in previous code):
    # - root_orient_6d (6)
    # - root_trans (3)
    # - partial_root_vel (4)?
    #
    # Let's just store the necessary global root info.
    # We can store:
    # - root_orient_6d (6)
    # - root_trans (3)
    # - root_vel_6d (6) -> Actually r_velocity approximates this, but 6D is better.
    # - root_trans_vel (3) -> l_velocity approximates this.
    #
    # The previous code stored 13 dims.
    # Let's store:
    # - root_orient_6d (6)
    # - root_trans (3)
    # - root_vel_6d (6)
    # Total 15 dims?
    #
    # If we stick to 13:
    # - root_orient_6d (6)
    # - root_trans (3)
    # - root_vel_6d (part of it?)
    #
    # Let's just store full root state (6+3+6+3 = 18) to be safe for perfect reconstruction?
    # Or just store what's needed to overwrite the integrated values.
    #
    # Let's store:
    # - root_orient_6d (6)
    # - root_trans (3)
    # - root_vel_6d (6)
    # - root_trans_vel (3)
    # Total 18.
    
    phys_13 = torch.cat([
        root_orient_6d, # 6
        root_trans,     # 3
        root_vel_6d,    # 6
        root_trans_vel  # 3
    ], dim=1) # 18 dims actually
    
    return base_263, phys_13

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--base_out', type=str, required=True)
    parser.add_argument('--phys_out', type=str, required=True)
    args = parser.parse_args()
    
    base, phys = vimogen_to_humanml3d_dual_stream(args.input)
    
    torch.save(base, args.base_out)
    torch.save(phys, args.phys_out)
    print(f"Saved base shape {base.shape} to {args.base_out}")
    print(f"Saved phys shape {phys.shape} to {args.phys_out}")
