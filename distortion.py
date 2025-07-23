import cv2
import numpy as np
import os
from nuscenes import NuScenes

nusc      = NuScenes(version='v1.0-trainval', dataroot='/home/yong/yongdb/dev/LLM-3D_Lane_Labeling_Machine/data/TCAR_DATA', verbose=False)
cam_token = 'CAM_FRONT'        # 저장하고 싶은 카메라
out_dir   = '/home/yong/yongdb/dev/LLM-3D_Lane_Labeling_Machine/data/TCAR_DATA/CAM_FRONT_UNDISTORTED'

os.makedirs(out_dir, exist_ok=True)

# ① 카메라 내부·외부 파라미터 가져오기
sample = nusc.sample[0]                            # 예시: 첫 sample
sd      = nusc.get('sample_data', sample['data'][cam_token])
calib   = nusc.get('calibrated_sensor', sd['calibrated_sensor_token'])

K = np.asarray(calib['camera_intrinsic']).astype(np.float64)
D = np.zeros((4,), np.float64)                     # NuScenes 카메라는 왜곡계수 제공 X → 0 으로

# ② 원본 이미지 로드
img_path = os.path.join(nusc.dataroot, sd['filename'])
img      = cv2.imread(img_path)
h, w     = img.shape[:2]

# ③ 매핑 테이블 생성 & 리맵
map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, (w, h), cv2.CV_16SC2)
img_ud     = cv2.remap(img, map1, map2, cv2.INTER_LINEAR)

# ④ 저장
out_path = os.path.join(out_dir, os.path.basename(img_path))
cv2.imwrite(out_path, img_ud)
print('saved:', out_path)