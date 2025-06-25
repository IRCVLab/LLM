import open3d as o3d
import numpy as np
import os
import sys

def read_and_display_pcd(pcd_path):
    """
    Read a PCD file and display its point coordinates
    
    Args:
        pcd_path (str): Path to the PCD file
    """
    # Check if file exists
    if not os.path.exists(pcd_path):
        print(f"Error: File not found: {pcd_path}")
        return
    
    try:
        # Read the point cloud
        pcd = o3d.io.read_point_cloud(pcd_path)
        
        if not pcd.has_points():
            print("Error: No points found in the PCD file")
            return
            
        # Get points as numpy array
        points = np.asarray(pcd.points)
        
        # Print basic information
        print(f"PCD File: {os.path.basename(pcd_path)}")
        print(f"Number of points: {len(points)}")
        print("\nFirst 10 points (x, y, z):")
        print("-" * 50)
        
        # Print first 10 points
        for i, point in enumerate(points[:10]):
            print(f"Point {i+1}: {point}")
            
        # Print statistics
        print("\nPoint Cloud Statistics:")
        print("-" * 50)
        print(f"X range: [{np.min(points[:, 0]):.4f}, {np.max(points[:, 0]):.4f}]")
        print(f"Y range: [{np.min(points[:, 1]):.4f}, {np.max(points[:, 1]):.4f}]")
        print(f"Z range: [{np.min(points[:, 2]):.4f}, {np.max(points[:, 2]):.4f}]")
        print(f"Mean position: [{np.mean(points[:, 0]):.4f}, {np.mean(points[:, 1]):.4f}, {np.mean(points[:, 2]):.4f}]")
        
    except Exception as e:
        print(f"Error reading PCD file: {str(e)}")

if __name__ == "__main__":
    pcd_path = './data_for_calib/ROI_PCD/000000.pcd'
    read_and_display_pcd(pcd_path)