
import torch
import numpy as np
import os
import sys
from scipy.spatial.transform import Rotation as R

# Rotation helpers (Same as convert)
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

def reconstruct_276(base_263, phys_18):
    """
    Reconstruct 276-dim from HumanML3D 263-dim + Phys Residual (18-dim).
    """
    T = base_263.shape[0]
    
    # 1. Extract Phys Residual
    root_orient_6d = phys_18[:, 0:6]
    root_trans = phys_18[:, 6:9]
    root_vel_6d = phys_18[:, 9:15]
    root_trans_vel = phys_18[:, 15:18]
    
    # 2. Extract Base Features (we only need the non-root parts if we use Phys for root)
    # But wait, 276 includes global joints positions/velocities.
    # We need to reconstruct global joints from ric_data and vel_data using Root info.
    
    ric_data = base_263[:, 4:67].reshape(T, 21, 3) # (T, 21, 3)
    rot_data = base_263[:, 67:193] # (T, 126) -> body_pose_6d
    vel_data = base_263[:, 193:259].reshape(T, 22, 3) # (T, 22, 3) -> joints_vel (local frame)
    
    # 3. Reconstruct Global Joints
    # Need Root Heading Rotation (Ry)
    root_rot_mat = rot6d_to_mat3x3(root_orient_6d) # (T, 3, 3)
    forward = root_rot_mat[:, :, 2]
    heading = torch.atan2(forward[:, 0], forward[:, 2]) # (T,)
    
    cos_h = torch.cos(heading)
    sin_h = torch.sin(heading)
    
    def rotate_y(vecs):
        # Rotate by +heading around Y (Inverse of rotate_y_inv)
        if vecs.dim() < 2: vecs = vecs.unsqueeze(0)
        x = vecs[..., 0]
        y = vecs[..., 1]
        z = vecs[..., 2]
        
        dims_to_add = x.dim() - 1
        shape_expand = [-1] + [1] * dims_to_add
        c = cos_h.view(*shape_expand)
        s = sin_h.view(*shape_expand)
        
        x_new = x * c - z * s
        y_new = y
        z_new = x * s + z * c
        return torch.stack([x_new, y_new, z_new], dim=-1)
    
    # ric_data is in local frame (root-relative, facing Z+)
    # Reconstruct Global Relative Position
    joints_rel_global = rotate_y(ric_data) # (T, 21, 3)
    
    # Add Root Position
    # joints include root at index 0
    # joints_rel_global corresponds to indices 1..21
    root_pos = root_trans.unsqueeze(1) # (T, 1, 3)
    
    # Construct full joints array (22, 3)
    # Index 0 is root_pos
    # Indices 1..21 are root_pos + joints_rel_global
    joints_global_body = root_pos + joints_rel_global
    joints_global = torch.cat([root_pos, joints_global_body], dim=1) # (T, 22, 3)
    
    # 4. Reconstruct Global Velocities
    # vel_data is in local frame
    joints_vel_global = rotate_y(vel_data) # (T, 22, 3)
    
    # 5. Assemble 276
    # 0:126   - body_pose_6d (rot_data)
    # 126:192 - joints (joints_global)
    # 192:258 - joints_vel (joints_vel_global)
    # 258:264 - root_orient_6d (from phys)
    # 264:270 - root_vel_6d (from phys)
    # 270:273 - root_trans (from phys)
    # 273:276 - root_trans_vel (from phys)
    
    rec_276 = torch.cat([
        rot_data,
        joints_global.reshape(T, -1),
        joints_vel_global.reshape(T, -1),
        root_orient_6d,
        root_vel_6d,
        root_trans,
        root_trans_vel
    ], dim=1)
    
    return rec_276

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_in', type=str, required=True)
    parser.add_argument('--phys_in', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading {args.base_in} and {args.phys_in}...")
    base = torch.load(args.base_in, map_location='cpu')
    phys = torch.load(args.phys_in, map_location='cpu')
    
    rec_276 = reconstruct_276(base, phys)
    
    print(f"Reconstructed shape: {rec_276.shape}")
    torch.save(rec_276, args.output)
    print(f"Saved to {args.output}")
