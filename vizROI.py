import open3d as o3d
import numpy as np
import os

# 1. PCD 파일 로드
pcd_file = "./data_for_calib/ROI_PCD/000000.pcd"  # 파일 경로를 변경해주세요
pcd = o3d.io.read_point_cloud(pcd_file)

# 2. 포인트 클라우드 정보 출력
print(f"로드된 포인트 수: {len(pcd.points)}")
print(f"X 범위: {np.min(pcd.points, axis=0)[0]:.2f} ~ {np.max(pcd.points, axis=0)[0]:.2f}")
print(f"Y 범위: {np.min(pcd.points, axis=0)[1]:.2f} ~ {np.max(pcd.points, axis=0)[1]:.2f}")
print(f"Z 범위: {np.min(pcd.points, axis=0)[2]:.2f} ~ {np.max(pcd.points, axis=0)[2]:.2f}")

# 3. 색상이 없으면 흰색으로 설정
if not pcd.has_colors():
    pcd.paint_uniform_color([1.0, 1.0, 1.0])

# 4. 시각화
o3d.visualization.draw_geometries([pcd],
                                window_name="viz",
                                width=1024,
                                height=768,
                                point_show_normal=False,
                                mesh_show_wireframe=False,
                                mesh_show_back_face=False)

