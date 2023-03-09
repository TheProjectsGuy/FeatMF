# Utilities for LFM and related things
"""
"""

# %%
import numpy as np
import einops as ein
import pandas as pd
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from typing import Union


# %%
# ---------------- Converters ----------------
# Transform (represent) for camera (image) frame to (in) agent frame
tf_agent_camera = np.array([
    [ 0, 1, 0, 0],
    [-1, 0, 0, 0],
    [ 0, 0, 1, 0],
    [ 0, 0, 0, 1]
])
"""
    Transformation matrix that represents the camera frame in the
    coordinates of the agent frame. Note that the convention of the
    agent frame is:
    - X is to the right
    - Y is up (-ve to gravity in up-is-up images)
    - Z is into the camera/agent frame (basically X x Y)
    
    The convention of the camera frame is:
    - X is down (+ve gravity in up-is-up images)
    - Y is to the right
    - Z is into the camera frame (out of image plane if viewing image)
    
    You can represent a camera frame point P in the agent frame using
    `P_camera = tf_agent_camera @ P_agent`
"""

# Pandas series to 4x4 TF
def pd_row_to_tf(row_series: pd.Series) -> np.ndarray:
    """
        Convert a single series (in a pandas dataframe) to homogeneous
        transformation matrix. The attributes ["Tx", "Ty", "Tz", "Qw", 
        "Qx", "Qy", "Qz"] should exist to get the pose from series.
        
        Parameters:
        - row_series:       A single item in the dataframe
        
        Returns:
        - tf:   A homogeneous transformation matrix of shape: [4, 4]
                and dtype float64.
    """
    r = row_series
    rot_m = R.from_quat([r.Qx, r.Qy, r.Qz, r.Qw]).as_matrix()
    trans_m = np.array([r.Tx, r.Ty, r.Tz]).reshape(3, 1)
    tf_mat = np.vstack([
            np.hstack([rot_m, trans_m]), 
            np.array([[0, 0, 0, 1]])])
    return tf_mat


# Open3D Point Cloud from Points and image
def pts_img_to_o3dpc(pts: np.ndarray, 
        img: Union[None,np.ndarray]=None) -> o3d.geometry.PointCloud:
    """
        Convert a point cloud to an Open3D point cloud. If an image is
        provided, the point cloud is also colored.
        
        Parameters:
        - pts:      Point cloud of shape [N, 3=XYZ] or [H, W, 3=XYZ].
        - img:      Image for coloring the point cloud. If None, the
                    point cloud is not colored. If not None, the shape
                    should be [H, W, 3=RGB] or [N, 3=RGB] and dtype
                    should be uint8.
        
        Returns:
        - pcd:      An Open3D point cloud object made with points and
                    colored using the image.
    """
    if len(pts.shape) == 3:
        pts = ein.rearrange(pts, "H W XYZ -> (H W) XYZ")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    if img is not None:
        if len(img.shape) == 3:
            img = ein.rearrange(img, "H W RGB -> (H W) RGB")
        pcd.colors = o3d.utility.Vector3dVector(img/255)
    return pcd

# %%
# --------------------- Photogrammetry ---------------------
# Get depth map using PNG image
def get_pcd_from_dimg(dimg: np.ndarray, pix_per_m: float=65535/10, 
            K=None, ret_hw: bool=True) -> np.ndarray:
    """
        Given a depth image, convert it to point cloud in camera's
        frame. Note that the axis convention is as follows
        - X: Vertically down the picture (+ve gravity in up-is-up 
            images)
        - Y: Horizontally to the right in picture
        - Z: Into the camera (out of image plane if viewing an image)
        
        Parameters:
        - dimg:         Depth image of shape [H, W] (int32)
        - pix_per_m:    Pixel value per meter (scale). If value 65535 
                        corresponds to 10 meters depth (in the `dimg`)
                        then this should be 65535/10.
        - K:            Camera intrinsic matrix. If None, the default
                        is the following \n
                        [   [ H/2,   0, H/2] \n
                            [   0, W/2, W/2] \n
                            [   0,   0,   1] ]
        - ret_hw:       If True, returned shape is [H, W, XYZ].
                        If False, returned shape is [H*W, XYZ].
        
        Returns:
        - pcd_np:       Points for point cloud. Shape is decided by
                        `ret_hw` parameter (either [H, W, XYZ] or 
                        [H*W, XYZ])
    """
    H, W = dimg.shape
    if K is None:
        K = np.array([[H/2, 0, H/2], [0, W/2, W/2], [0, 0, 1]], 
                        dtype=float)
    # Convert depth PNG (pixel values) to metric (meters)
    depth_m = np.array(dimg) * (1/pix_per_m)
    # [H, W, 3=[x, y, 1]] tensor for homogeneous pixel coordinates
    yv, xv = np.meshgrid(np.arange(W), np.arange(H))
    xy1 = np.stack([xv, yv, np.ones_like(xv)], axis=-1)
    hom_pts = ein.rearrange(xy1, "H W uvw -> uvw (H W)")  # 3, N
    # Get the point in the 3D world using depth
    wpts = np.linalg.inv(K) @ hom_pts
    wpts = ein.rearrange(wpts, "XYH (H W) -> H W XYH", H=H, W=W)
    wpts /= wpts[..., [2]]  # Homogenize last dim to 1
    wpts_m = depth_m[..., np.newaxis] * wpts    # In 'm'
    wpts_m[..., -1] *= -1   # Depth (Z) is -ve: Z is into camera
    pts_m = wpts_m  # As [H, W, XYZ]
    if not ret_hw:  # As [H*W, XYZ]
        pts_m = ein.rearrange(pts_m, "H W XYZ -> (H W) XYZ")
    return pts_m
