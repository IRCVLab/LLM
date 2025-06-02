import numpy as np
import cv2 
import open3d as o3d
import matplotlib.pyplot as plt
import io


def sample_lane_points(points_3d, num_samples=20):
    points = points_3d.T
    if points.shape[0] < num_samples:
        return points

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
    
    sampled = points[sampled_indices]
    
    return sampled


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

def lane_points_3d_from_pcd_and_lane(
    rgb_image, 
    point_cloud, 
    k, 
    r_lidar_to_camera_coordinate, 
    t_lidar_to_camera_coordinate, 
    lane_polyline_img_coords,
    lane_thickness=5):

    t_lidar_to_camera_coordinate = t_lidar_to_camera_coordinate.reshape(-1, 1) / 1000.0

    points_in_front_of_lidar = []
    for point_i in point_cloud:
        if point_i[0] >= 0:
            points_in_front_of_lidar.append(point_i)
    point_cloud = np.array(points_in_front_of_lidar)
    
    points_in_camera_coordinate = np.dot(r_lidar_to_camera_coordinate, point_cloud.T) + t_lidar_to_camera_coordinate

    points_in_image = np.dot(k, points_in_camera_coordinate)
    points_in_image = points_in_image / points_in_image[2, :]
    points_in_image = points_in_image[0:2, :]
    points_in_image = points_in_image.T


    fused_points = []

    for i, point_img in enumerate(points_in_image):
        if (0 <= point_img[0] < rgb_image.shape[1]) and (0 <= point_img[1] < rgb_image.shape[0]):
            fused_points.append((point_img[0], point_img[1], points_in_camera_coordinate[2, i]))

    fused_points = np.array(fused_points)

    lane_mask = create_lane_mask(rgb_image.shape, lane_polyline_img_coords, thickness=lane_thickness)
    lane_indices = filter_points_on_lane(fused_points, lane_mask)
    lane_points_3d = fused_points[lane_indices]

    points_in_image = lane_points_3d[:,:2]
    depth_inside_image = lane_points_3d[:,2]

    homo = np.ones((points_in_image.shape[0], 1))
    points_in_image_homo = np.concatenate([points_in_image, homo], axis=1)
    s = np.expand_dims(depth_inside_image, axis=0)
    points_in_camera_homo = np.linalg.inv(k) @ points_in_image_homo.T * s
    points_in_lidar_homo = np.linalg.inv(r_lidar_to_camera_coordinate) @ (points_in_camera_homo - t_lidar_to_camera_coordinate)

    filtered_indices = []
    for idx in range(points_in_lidar_homo.shape[1]):
        x = points_in_lidar_homo[0, idx]
        y = points_in_lidar_homo[1, idx]
        if abs(x) > 1.0 and abs(y) > 1.0:
            filtered_indices.append(idx)
    
    points_in_lidar_homo = points_in_lidar_homo[:, filtered_indices]

    print(f"mapping된 point 개수: {len(points_in_lidar_homo[0])}")
    return points_in_lidar_homo 



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
    # fig = plt.figure()
    # plt.imshow(rgb_image)
    # plt.scatter(points_in_image[:, 0].tolist(), points_in_image[:, 1].tolist(), c=depth_inside_image, s=3)
    # ax = plt.gca()
    # ax.axes.xaxis.set_ticks([])
    # ax.axes.yaxis.set_ticks([])
    # img_lidar_points = get_img_from_fig(fig=fig, dpi=500)

