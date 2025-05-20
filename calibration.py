# pcd_mapping.py
import numpy as np
from scipy.spatial import cKDTree
import open3d as o3d
import json

def map_lane_to_pcd(pcd, lane_uv, pixel_thresh=2.0):

    with open("cal_parameter.json", 'r', encoding='utf-8') as f:
        C = json.load(f)

    extrinsic = np.array(C['extrinsic'], dtype=float)
    intrinsic = np.array(C['intrinsic'], dtype=float)
    
    pts = np.asarray(pcd.points)

    ones = np.ones((len(pts),1))
    pts_h = np.hstack([pts, ones])
    cam_h = (extrinsic @ pts_h.T).T
    cam = cam_h[cam_h[:,2] > 0, :3]

    proj = intrinsic @ cam.T
    proj /= proj[2:3,:]
    uv_all = proj[:2,:].T

    tree = cKDTree(lane_uv)
    dists, idxs = tree.query(uv_all, k=1)
    mask = dists < pixel_thresh

    selected_pts = pts[cam[:,2]>0][mask]
    return selected_pts
