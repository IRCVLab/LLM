import numpy as np
import cv2 
import open3d as o3d
import matplotlib.pyplot as plt
import io


def sample_lane_points(points_3d, points_2d, num_samples=20):
    points = points_3d.T
    if points.shape[0] < num_samples:
        return points, points_2d

    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumdist = np.insert(np.cumsum(dists), 0, 0)
    target_dists = np.linspace(0, cumdist[-1], num_samples)

    sampled_indices = []
    j = 0
    for td in target_dists:
        while j < len(cumdist)-1 and cumdist[j+1] < td:
            j += 1
        if j >= len(points):
            break
        sampled_indices.append(j)
    
    sampled_indices = list(dict.fromkeys(sampled_indices))
    while len(sampled_indices) < num_samples and len(points) > 0:
        sampled_indices.append(len(points) - 1)
    
    sampled_indices = sampled_indices[:num_samples]
    
    sampled_3d = points[sampled_indices]
    sampled_2d = points_2d[sampled_indices]
    
    return sampled_3d, sampled_2d


def load_calibration_params():
    k = np.load("./calibration/k.npy")
    r = np.load("./calibration/r.npy")
    t = np.load("./calibration/t.npy")
    distortion = np.load(f"./calibration/distortion.npy")
    return t, r, k, distortion

def create_lane_mask(image_shape, lane_points, thickness):

    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    pts = np.array(lane_points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(mask, [pts], isClosed=False, color=1, thickness=thickness)
    # plt.imshow(mask, cmap='gray')
    # plt.title('Lane Mask')
    # plt.show()
    return mask

def filter_points_on_lane(points_in_image, lane_mask):

    indices = []
    for i, (x, y, z) in enumerate(points_in_image.astype(int)):
        if 0 <= x < lane_mask.shape[1] and 0 <= y < lane_mask.shape[0]:
            if lane_mask[y, x] > 0:
                indices.append(i)
    return indices

def filter_points_in_front(point_cloud):
    points_in_front_of_lidar = []
    for point_i in point_cloud:
        if point_i[0] >= 0:
            points_in_front_of_lidar.append(point_i)
    
    return np.array(points_in_front_of_lidar)

def transform_to_camera(points_in_front, r, t):
    return np.dot(r, points_in_front.T) + t

def project_to_image(points_in_camera_coordinate, k):
    points_in_image = np.dot(k, points_in_camera_coordinate)
    points_in_image = points_in_image / points_in_image[2, :]
    points_in_image = points_in_image[0:2, :]
    points_in_image = points_in_image.T

    return points_in_image

def create_fused_points(points_in_image, points_in_camera_coordinate, rgb_image):
    fused_points = []

    for i, point_img in enumerate(points_in_image):
        if (0 <= point_img[0] < rgb_image.shape[1]) and (0 <= point_img[1] < rgb_image.shape[0]):
            fused_points.append((point_img[0], point_img[1], points_in_camera_coordinate[2, i]))

    return np.array(fused_points)

def convert_to_homogeneous(points):
    homo = np.ones((points.shape[0], 1))
    return np.concatenate([points, homo], axis=1)

def convert_to_camera(points_homo, k, depth):
    s = np.expand_dims(depth, axis=0)
    return np.linalg.inv(k) @ points_homo.T * s

def convert_to_lidar(points_camera, r, t):
    return np.linalg.inv(r) @ (points_camera - t)

def point_masking(lane_polyline_img_coords, fused_points):
    # fused_points: (N, 3) [x_img, y_img, z]
    points_in_image = fused_points[:, :2]
    if isinstance(lane_polyline_img_coords, np.ndarray):
        coords = lane_polyline_img_coords
    else:
        coords = np.array(lane_polyline_img_coords)
    if coords.shape[0] == 1:
        if fused_points.shape[0] == 0:
            return np.zeros((0, 3))
        click_xy = coords[0]
        dists = np.linalg.norm(points_in_image - click_xy, axis=1)
        idx = np.argmin(dists)
        closest_3d = fused_points[idx, :3]  # (3,) shape
        return closest_3d


########################
# image to point cloud #
########################
def projection_img_to_pcd(
    rgb_image, 
    point_cloud, 
    k, 
    r_lidar_to_camera_coordinate, 
    t_lidar_to_camera_coordinate, 
    lane_polyline_img_coords,
    lane_thickness=5,
    single_click=False):

    t_lidar_to_camera_coordinate = t_lidar_to_camera_coordinate.reshape(-1, 1) / 1000.0

    # front points만 필터링
    points_in_front = filter_points_in_front(point_cloud)

    # 카메라 좌표계로 변환
    points_in_camera_coordinate = transform_to_camera(points_in_front, r_lidar_to_camera_coordinate, t_lidar_to_camera_coordinate)

    # 이미지 좌표로 프로젝션
    points_in_image = project_to_image(points_in_camera_coordinate, k)
    
    # 이미지 범위 내의 점들만 필터링 + 반환 값은 (x, y, depth)
    fused_points = create_fused_points(points_in_image, points_in_camera_coordinate, rgb_image)

    # 포인트 변환
    if single_click:
        lane_points_3d = point_masking(lane_polyline_img_coords, fused_points)
    # lane points 변환
    else:
        lane_mask = create_lane_mask(rgb_image.shape, lane_polyline_img_coords, thickness=lane_thickness)
        lane_indices = filter_points_on_lane(fused_points, lane_mask)
        lane_points_3d = fused_points[lane_indices]


    if lane_points_3d.ndim == 1:
        # 단일 점 (3,)
        points_in_image = lane_points_3d[:2].reshape(1, 2)
        depth_inside_image = np.array([lane_points_3d[2]])
    elif lane_points_3d.ndim == 2:
        # 여러 점 (N, 3)
        points_in_image = lane_points_3d[:, :2]
        depth_inside_image = lane_points_3d[:, 2]
    else:
        # 예외 처리
        points_in_image = np.zeros((0, 2))
        depth_inside_image = np.zeros((0,))

    # 이미지 좌표를 homogeneous coordinate로 변환
    points_in_image_homo = convert_to_homogeneous(points_in_image)

    # 깊이 정보를 사용하여 카메라 좌표계로 변환
    points_in_camera_homo = convert_to_camera(points_in_image_homo, k, depth_inside_image)
    # 카메라 좌표계를 LiDAR 좌표계로 변환
    points_in_lidar_homo = convert_to_lidar(points_in_camera_homo, r_lidar_to_camera_coordinate, t_lidar_to_camera_coordinate)

    filtered_indices = []
    for idx in range(points_in_lidar_homo.shape[1]):
        x = points_in_lidar_homo[0, idx]
        y = points_in_lidar_homo[1, idx]
        # if abs(x) > 1.0 and abs(y) > 1.0:
        filtered_indices.append(idx)
    
    points_in_lidar_homo = points_in_lidar_homo[:, filtered_indices]
    points_in_image = points_in_image[filtered_indices]
    if single_click:
        points_in_lidar_homo = points_in_lidar_homo.reshape(-1)
    return points_in_lidar_homo, points_in_image 



########################
# point cloud to image #
######################## 
def projection_pcd_to_img(
    lane_polyline_lidar,  # (N, 3) ndarray
    k,                    # (3, 3) camera intrinsic
    r_lidar_to_camera,    # (3, 3) extrinsic rotation
    t_lidar_to_camera,    # (3,) extrinsic translation
    img_shape=None,        # (H, W, 3) or (H, W)
    single_click=False
):
    # 1. LiDAR → Camera 좌표계 변환
    lane_polyline_lidar = np.asarray(lane_polyline_lidar)

    if single_click:
        # 입력이 (3,) 또는 (1,3)일 때도 항상 (1,3)으로 맞춤
        if lane_polyline_lidar.ndim == 1 and lane_polyline_lidar.shape[0] == 3:
            lane_polyline_lidar = lane_polyline_lidar.reshape(1, 3)
        elif lane_polyline_lidar.ndim == 2 and lane_polyline_lidar.shape[0] == 3 and lane_polyline_lidar.shape[1] != 3:
            lane_polyline_lidar = lane_polyline_lidar.T
        # 변환
        t_lidar_to_camera = t_lidar_to_camera.reshape(-1, 1) / 1000.0
        lane_polyline_cam = transform_to_camera(lane_polyline_lidar, r_lidar_to_camera, t_lidar_to_camera)  # (3, 1)
        lane_polyline_img = project_to_image(lane_polyline_cam, k)  # (1, 2)
        z_cam = lane_polyline_cam[2, :]
        mask = z_cam > 0.01
        lane_polyline_img = lane_polyline_img[mask]
        if img_shape is not None and lane_polyline_img.shape[0] > 0:
            H, W = img_shape[:2]
            mask2 = (lane_polyline_img[:,0] >= 0) & (lane_polyline_img[:,0] < W) & \
                    (lane_polyline_img[:,1] >= 0) & (lane_polyline_img[:,1] < H)
            lane_polyline_img = lane_polyline_img[mask2]
        if lane_polyline_img.ndim != 2 or lane_polyline_img.shape[0] < 1:
            print("[projection_pcd_to_img] returning empty (0,2) array!")
            return np.empty((0, 2))
        return lane_polyline_img
    else:
        t_lidar_to_camera = t_lidar_to_camera.reshape(-1, 1) / 1000.0
        lane_polyline_cam = transform_to_camera(lane_polyline_lidar, r_lidar_to_camera, t_lidar_to_camera)  # (3, N)
        # 2. Camera → Image projection
        lane_polyline_img = project_to_image(lane_polyline_cam, k)  # (N, 2)
        # 3. 깊이(z)가 0 이하인 점 제외
        z_cam = lane_polyline_cam[2, :]
        mask = z_cam > 0.01
        lane_polyline_img = lane_polyline_img[mask]
        # 4. 이미지 범위 내로 클리핑 (mask2) 제거! 이미지 바깥 점도 모두 포함해서 반환
        if img_shape is not None and lane_polyline_img.shape[0] > 0:
            H, W = img_shape[:2]
            mask2 = (lane_polyline_img[:,0] >= 0) & (lane_polyline_img[:,0] < W) & \
                    (lane_polyline_img[:,1] >= 0) & (lane_polyline_img[:,1] < H)
            lane_polyline_img = lane_polyline_img[mask2]
        if lane_polyline_img.ndim != 2 or lane_polyline_img.shape[0] < 2:
            return np.empty((0, 2))
        return lane_polyline_img


def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def lidar_points_in_image(
    rgb_image, 
    point_cloud,
    k,
    r_lidar_to_camera_coordinate,
    t_lidar_to_camera_coordinate, 
):
    t_lidar_to_camera_coordinate = t_lidar_to_camera_coordinate.reshape(-1, 1) / 1000

    # keep points that are in front of LiDAR
    points_in_front_of_lidar = []
    for point_i in point_cloud:
        if point_i[0] >= 0:
            points_in_front_of_lidar.append(point_i)
    point_cloud = np.array(points_in_front_of_lidar)
    
    # translate lidar points to camera coordinate system
    points_in_camera_coordinate = np.dot(r_lidar_to_camera_coordinate, point_cloud.T) + t_lidar_to_camera_coordinate

    # project points form camera coordinate to image
    points_in_image = np.dot(k, points_in_camera_coordinate)
    points_in_image = points_in_image / points_in_image[2, :]
    points_in_image = points_in_image[0:2, :]
    points_in_image = points_in_image.T
    
    # keep points that are inside image
    points_inside_image = []
    depth_inside_image = []
    
    for i, point_i in enumerate(points_in_image):
        if (0 <= point_i[0] < rgb_image.shape[1]) and (0 <= point_i[1] < rgb_image.shape[0]):
            points_inside_image.append(point_i)
            depth_inside_image.append(points_in_camera_coordinate[2, i])
    
    points_in_image = np.array(points_inside_image)
    depth_inside_image = np.array(depth_inside_image)
    
    homo = np.ones((points_in_image.shape[0], 1))
    points_in_image_homo = np.concatenate([points_in_image, homo], axis=1)
    s = np.expand_dims(depth_inside_image, axis=0)
    points_in_camera_homo = np.linalg.inv(k) @ points_in_image_homo.T * s
    points_in_lidar_homo = np.linalg.inv(r_lidar_to_camera_coordinate) @ (points_in_camera_homo - t_lidar_to_camera_coordinate)

    return points_in_lidar_homo
