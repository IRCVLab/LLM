import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
import os 
import vtk
from utils.converter import load_calibration_params, projection_pcd_to_img, projection_img_to_pcd
import matplotlib.pyplot as plt
import open3d as o3d
class EventTools:
    """
    Mixin for event handling methods (mouse, keyboard, matplotlib, etc)
    Assumes main Window class initializes all shared state and UI components.
    """
    
    def initialize_lane_structures(self):
        """초기화 시 통합 레인 데이터 구조 설정"""
        # 통합 레인 리스트 - 이미지와 VTK 레인 데이터를 함께 관리
        if not hasattr(self, 'unified_lanes'):
            '''
            [{
            'img_curve': curve_artist, 
            'img_points': [artists], 
            'vtk_actor': actor, 
            'vtk_point_actors': [actors], 
            'type': lane_type, 
            'points_3d': [...], 
            'points_2d': [...]
            }]
            '''
            self.unified_lanes = []  
        
        # 현재 작업 중인 레인 버퍼
        if not hasattr(self, 'current_lane_buffer'):
            self.current_lane_buffer = {
                'img_points': [],   
                'img_artists': [], 
                'vtk_points': [],   
                'vtk_actors': [],   
                'type': None        
            }
            

    def on_mpl_click(self, event):
        # Matplotlib canvas click event
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        # 2D 점 저장
        self.lane_points.append((event.xdata, event.ydata))
        lane_type = None
        if self.beforeRadioChecked == self.lineRadio1:
            lane_type = 'Yellow'
        elif self.beforeRadioChecked == self.lineRadio2:
            lane_type = 'White'
        elif self.beforeRadioChecked == self.lineRadio3:
            lane_type = 'WhiteDash'
        lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
        edgecolor = lane_info['mpl_color']
        # 이미지에 점 시각화
        artist = self.axes.scatter(
            event.xdata, event.ydata,
            s=60, facecolors='white', edgecolors=edgecolor, linewidths=2
        )
        self.lane_point_artists.append(artist)
        self.all_point_artists.append(artist)
        self.canvas.draw()
        # --- 2D→3D 변환 및 VTK에 시각화 ---
        try:
            if hasattr(self, "calibration_params"):
                t, r, k, _ = self.calibration_params
            else:
                t, r, k, _ = load_calibration_params()
                self.calibration_params = (t, r, k, _)

                
            # point cloud 준비
            img_file = self.list_img_path[self.imgIndex]
            pcd_file = os.path.splitext(os.path.basename(img_file))[0] + '.pcd'
            pcd_path = os.path.join(self.pcd_dir, pcd_file)
            if not os.path.exists(pcd_path):
                print(f"PCD file not found: {pcd_path}")
                return
            pcd = o3d.t.io.read_point_cloud(pcd_path)
            np_points = pcd.point.positions.numpy()

            rgb_image = self.pyt.get_array() if hasattr(self, 'pyt') else None
            points_2d = np.array([[event.xdata, event.ydata]])
            points_3d, _ = projection_img_to_pcd(
                rgb_image, np_points, k, r, t, points_2d, single_click=True
            )
            print(f"[on_mpl_click] shape: {points_3d.shape}")
            self.addLanePointsToVTK(points_3d, color=lane_info['vtk_color'], size=7, single_click=True)
            if not hasattr(self, 'vtk_lane_points'):
                self.vtk_lane_points = []
            self.vtk_lane_points.append(points_3d)
        except Exception as e:
            print(f"[on_mpl_click] error: {e}")



    def on_vtk_click(self, obj, event):
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        ctrl = interactor.GetControlKey()
        shift = interactor.GetShiftKey()
        alt = interactor.GetAltKey() if hasattr(interactor, "GetAltKey") else False
        if ctrl or shift or alt:
            return
        x, y = interactor.GetEventPosition()
        renderer = self.vtkRenderer

        picker = vtk.vtkPointPicker()
        picker.Pick(x, y, 0, renderer)

        if not (hasattr(self, 'pcd_points_np') and hasattr(self, 'pcd_colors_np')):
            try:
                pcd_file = self.list_pcd_path[self.imgIndex]
                pcd_path = os.path.join(self.pcd_dir, pcd_file)
                import open3d as o3d
                pcd = o3d.t.io.read_point_cloud(pcd_path)
                points_np = pcd.point.positions.numpy()
                colors_np = pcd.point.colors.numpy() if "colors" in pcd.point else None
                mask = ~np.isnan(points_np).any(axis=1)

                points_np = points_np[mask]
                if colors_np is not None:
                    colors_np = colors_np[mask]
                self.pcd_points_np = points_np
                self.pcd_colors_np = colors_np
            except Exception as e:
                print(f"[VTK Click] Could not load point cloud: {e}")
                return

        if self.pcd_colors_np is not None:
            color_norm = np.linalg.norm(self.pcd_colors_np, axis=1)
            color_mask = color_norm > 100
            filtered_points = self.pcd_points_np[color_mask]
        else:
            filtered_points = self.pcd_points_np

        min_dist = float('inf')
        nearest = None
        renderer = self.vtkRenderer
        for pt in filtered_points:
            renderer.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
            renderer.WorldToDisplay()
            display_coords = renderer.GetDisplayPoint()
            dx = display_coords[0] - x
            dy = display_coords[1] - y
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                nearest = pt

        # --- Accumulate clicked points for VTK polyline ---
        if not hasattr(self, 'vtk_lane_points'):
            self.vtk_lane_points = []
        self.vtk_lane_points.append(nearest)
        # VTK 점 시각화
        lane_type = None
        if self.beforeRadioChecked == self.lineRadio1:
            lane_type = 'Yellow'
        elif self.beforeRadioChecked == self.lineRadio2:
            lane_type = 'White'
        elif self.beforeRadioChecked == self.lineRadio3:
            lane_type = 'WhiteDash'
        lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
        vtk_color = lane_info['vtk_color']
        self.addLanePointsToVTK(points=np.array([nearest]), color=vtk_color, size=7, single_click=True)
        # --- 3D→2D 변환 및 이미지에 시각화 ---
        try:
            if hasattr(self, "calibration_params"):
                t, r, k, _ = self.calibration_params
            else:
                t, r, k, _ = load_calibration_params()
                self.calibration_params = (t, r, k, _)
            img_shape = self.img.shape if hasattr(self, "img") else None
            points_2d = projection_pcd_to_img(
                np.array([nearest]), k, r, t, img_shape, single_click=True
            )
            print(f"[on_vtk_click] shape: {points_2d.shape}")
            if points_2d is not None and len(points_2d) > 0:
                # 산점도 표시
                artist = self.axes.scatter(
                    points_2d[0][0], points_2d[0][1],
                    s=60, facecolors='white', edgecolors=lane_info['mpl_color'], linewidths=2
                )
                self.lane_point_artists.append(artist)
                self.all_point_artists.append(artist)
                self.canvas.draw()

                # Add to lane points buffer
            if not hasattr(self, 'lane_points'):
                self.lane_points = []
            self.lane_points.append([float(points_2d[0][0]), float(points_2d[0][1])])
        except Exception as e:
            print(f"[on_vtk_click] error: {e}")



    def radioButtonClicked(self):
        self.beforeRadioChecked.setAutoExclusive(False)
        self.beforeRadioChecked.setChecked(False)
        self.beforeRadioChecked.setAutoExclusive(True)

        if self.lineRadio1.isChecked():
            self.beforeRadioChecked = self.lineRadio1
        elif self.lineRadio2.isChecked():
            self.beforeRadioChecked = self.lineRadio2
        elif self.lineRadio3.isChecked():
            self.beforeRadioChecked = self.lineRadio3
        else:
            print("outofrange at radiobutton")

    def delAllLine(self):
        ''' del all lines to figure '''
        self.butdisconnect()
        while self.list_points:
            if self.list_points:
                self.list_points[-1].line.remove()
                self.list_points.pop()
            if self.axes.patches:
                self.axes.patches[-1].remove()
            if self.list_points_type:
                self.list_points_type.pop()
        self.butconnect()

    def delete_last_lane(self):
        """통합 레인 삭제 함수 - 이미지와 VTK 모두에서 마지막 레인 삭제"""
        if not hasattr(self, 'unified_lanes') or not self.unified_lanes:
            return
            
        # 마지막 통합 레인 가져오기
        last_lane = self.unified_lanes.pop()
        
        # 1. 이미지(matplotlib) 레인 삭제
        if 'img_curve' in last_lane and last_lane['img_curve'] is not None:
            last_lane['img_curve'].remove()
            
        # 2. 이미지 점들 삭제
        if 'img_points' in last_lane and last_lane['img_points']:
            for artist in last_lane['img_points']:
                if hasattr(self, 'all_point_artists') and artist in self.all_point_artists:
                    self.all_point_artists.remove(artist)
                if hasattr(self, 'lane_point_artists') and artist in self.lane_point_artists:
                    self.lane_point_artists.remove(artist)
                artist.remove()
        
        # 3. VTK 레인 액터 삭제
        if 'vtk_actor' in last_lane and last_lane['vtk_actor'] is not None:
            self.vtkRenderer.RemoveActor(last_lane['vtk_actor'])
            # lane_vtk_actors에서도 제거
            if hasattr(self, 'lane_vtk_actors') and last_lane['vtk_actor'] in self.lane_vtk_actors:
                self.lane_vtk_actors.remove(last_lane['vtk_actor'])
            
        # 4. VTK 점 액터들 삭제
        if 'vtk_point_actors' in last_lane and last_lane['vtk_point_actors']:
            for act in last_lane['vtk_point_actors']:
                self.vtkRenderer.RemoveActor(act)
                if hasattr(self, 'lane_vtk_actors') and act in self.lane_vtk_actors:
                    self.lane_vtk_actors.remove(act)
        
        # 5. 관련 데이터 구조 업데이트
        if hasattr(self, 'lane_labels') and self.lane_labels and 'lane_label_idx' in last_lane:
            # 이미지 레인 라벨 정보 제거
            for i in range(len(self.lane_labels)-1, -1, -1):
                if self.lane_labels[i] == last_lane['lane_label_idx']:
                    self.lane_labels.pop(i)
                    break
                    
        if hasattr(self, 'lane_curve_artists') and self.lane_curve_artists and 'img_curve' in last_lane:
            # 이미지 커브 아티스트 제거
            if last_lane['img_curve'] in self.lane_curve_artists:
                self.lane_curve_artists.remove(last_lane['img_curve'])
                
        if hasattr(self, 'vtk_lanes') and self.vtk_lanes and 'vtk_points' in last_lane:
            # VTK 레인 포인트 제거
            for i in range(len(self.vtk_lanes)-1, -1, -1):
                if isinstance(last_lane['vtk_points'], list) and isinstance(self.vtk_lanes[i], list):
                    # 두 리스트를 numpy 배열로 변환하여 비교
                    if np.array_equal(np.array(self.vtk_lanes[i]), np.array(last_lane['vtk_points'])):
                        self.vtk_lanes.pop(i)
                        break
                elif np.array_equal(self.vtk_lanes[i], last_lane['vtk_points']):
                    self.vtk_lanes.pop(i)
                    break
        
        # 화면 갱신
        self.canvas.draw()
        if hasattr(self, 'vtkWidget'):
            self.vtkWidget.GetRenderWindow().Render()
            

    def delete_point(self):
        """현재 작업 중인 점 삭제 - 이미지와 VTK 모두 고려"""
        # 이미지 점 삭제
        self.delete_img_point()
        # VTK 점 삭제
        self.delete_vtk_point()
        
    def delete_img_point(self):
        """이미지에서 마지막 점 삭제 (현재 그리는 중인 레인의 점)"""
        # 가장 최근 레인의 마지막 점만 삭제
        if hasattr(self, 'lane_points') and self.lane_points:
            self.lane_points.pop()
        if hasattr(self, 'lane_point_artists') and self.lane_point_artists:
            last_artist = self.lane_point_artists.pop()
            last_artist.remove()
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        


    def delete_vtk_point(self):
        """VTK에서 마지막 점 삭제 (현재 그리는 중인 레인의 점)"""
        import numpy as np
        # 가장 최근 레인(폴리라인) 구조에서 마지막 점 제거
        if hasattr(self, 'vtk_lanes') and self.vtk_lanes:
            last_lane = self.vtk_lanes[-1]
            # numpy 배열인지, 파이썬 리스트인지 구분
            if isinstance(last_lane, np.ndarray):
                if last_lane.shape[0] > 0:
                    # 배열에서 마지막 행 제거
                    self.vtk_lanes[-1] = last_lane[:-1]
                # 비어 있으면 레인 자체 제거
                if self.vtk_lanes[-1].shape[0] == 0:
                    self.vtk_lanes.pop()
            elif isinstance(last_lane, list):
                if last_lane:
                    last_lane.pop()
                if not last_lane:
                    self.vtk_lanes.pop()
        elif hasattr(self, 'vtk_lane_points') and self.vtk_lane_points:
            # 아직 폴리라인으로 확정 전인 클릭 버퍼에서 pop
            self.vtk_lane_points.pop()
        if hasattr(self, 'lane_vtk_actors') and self.lane_vtk_actors:
            actor = self.lane_vtk_actors.pop()
            self.vtkRenderer.RemoveActor(actor)
        if hasattr(self, 'vtkWidget'):
            self.vtkWidget.GetRenderWindow().Render()
            

    def add_lane(self):
        """Collect current buffers and visualize lane using lane.py helpers only"""
        # Ensure lane infrastructure exists
        if not hasattr(self, 'unified_lanes'):
            self.unified_lanes = []
        if not hasattr(self, 'current_lane_buffer'):
            self.initialize_lane_structures()

        # Determine lane type
        lane_type = None
        if self.beforeRadioChecked == self.lineRadio1:
            lane_type = 'Yellow'
        elif self.beforeRadioChecked == self.lineRadio2:
            lane_type = 'White'
        elif self.beforeRadioChecked == self.lineRadio3:
            lane_type = 'WhiteDash'

        # Gather 2D points
        points_2d = self.lane_points.copy() if hasattr(self, 'lane_points') else []

        # Gather and filter 3D points (skip ego/zero)
        points_3d = []
        if hasattr(self, 'vtk_lane_points'):
            for pt in self.vtk_lane_points:
                arr = np.asarray(pt).reshape(-1)
                if arr.shape == (3,) and not np.allclose(arr[:2], 0, atol=1e-3):
                    points_3d.append(arr.tolist())

        # Collect click-created actors before drawing unified lane
        img_click_actors = self.lane_point_artists.copy() if hasattr(self, 'lane_point_artists') else []
        num_vtk_click_pts = len(self.vtk_lane_points) if hasattr(self, 'vtk_lane_points') else 0
        vtk_click_actors = []
        if num_vtk_click_pts and hasattr(self, 'lane_vtk_actors'):
            vtk_click_actors = self.lane_vtk_actors[-num_vtk_click_pts:]

        # Build lane_data dict expected by draw_lane_from_unified
        lane_data = {
            'type': lane_type,
            'points_2d': points_2d,
            'points_3d': points_3d,
            'img_points': img_click_actors,
            'vtk_point_actors': vtk_click_actors,
        }

        # Use lane.py helper to draw
        lane_data = self.draw_lane_from_unified(lane_data)

        # draw_lane_from_unified may have appended new scatter/actors; merge lists to ensure all are tracked
        if 'img_points' in lane_data:
            lane_data['img_points'] = list(set(lane_data['img_points'] + img_click_actors))
        if 'vtk_point_actors' in lane_data:
            lane_data['vtk_point_actors'] = list(set(lane_data['vtk_point_actors'] + vtk_click_actors))

        # Store
        self.unified_lanes.append(lane_data)

        # Clear buffers for next lane
        self.lane_points = []
        self.lane_point_artists = []
        self.vtk_lane_points = []

        # Refresh views
        self.canvas.draw()
        if hasattr(self, 'vtkWidget'):
            self.vtkWidget.GetRenderWindow().Render()
    
    def butconnect(self):
        for pts in self.list_points:
            pts.connect()
        self.canvas.draw()

    def butdisconnect(self):
        for pts in self.list_points:
            pts.disconnect()
        self.canvas.draw()
    

    def loadNextImage(self):
        self.saveAll(self.img_path)
        self.plotBackGround(self.img_path,0)

    def loadPrevImage(self):
        self.saveAll(self.img_path)
        self.plotBackGround(self.img_path,1)

    def msgBoxReachEdgeEvent(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setWindowTitle('WARNING')
        msgBox.setText( "Reach the end of image" )
        msgBox.setInformativeText( "Press OK to continue" )
        msgBox.addButton( QMessageBox.Ok )

        msgBox.setDefaultButton( QMessageBox.Ok )
        ret = msgBox.exec_()

        if ret == QMessageBox.Ok:
            return True
        else:
            return False

    def saveAll(self, img_path):
        ind = self.imgIndex
        if self.imgIndex == len(self.list_img_path):
            ind = ind - 1

        if self.imgIndex == -1:
            ind = ind + 1

        # Load calibration parameters
        t, r, k, distortion = load_calibration_params()
        
        # Get current image path
        img_file = self.list_img_path[ind]
        file_path = os.path.join(img_path, 'image', img_file)
        
        # Prepare lane data
        lane_data = []
        
        # 통합 레인 데이터 구조에서 저장
        if hasattr(self, 'unified_lanes') and self.unified_lanes:
            for lane in self.unified_lanes:
                lane_type = lane.get('type', 'Default')
                
                # Convert lane type to category using LANE_COLORS
                lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
                category = lane_info['category']
                
                # 3D 점 좌표 (VTK 점)
                xyz = None
                if 'points_3d' in lane and lane['points_3d'] is not None:
                    xyz = np.array(lane['points_3d']).T.tolist()
                elif 'vtk_points' in lane and lane['vtk_points']:
                    xyz = np.array(lane['vtk_points']).T.tolist()
                
                # 2D 점 좌표 (이미지 점)
                uv = None
                if 'points_2d' in lane and lane['points_2d'] is not None:
                    uv = np.array(lane['points_2d']).T.tolist()
                
                # 둘 다 있는 경우만 저장
                if xyz and uv:
                    # Create lane line data
                    lane_line = {
                        "category": category,
                        "visibility": [1.0] * len(xyz[0]),  # All points are visible
                        "uv": uv,
                        "xyz": xyz
                    }
                    lane_data.append(lane_line)
        
        # 기존 방식 (이전 버전 호환성)
        elif hasattr(self, 'sampled_pts') and self.sampled_pts is not None:
            # Get lane type from UI selection
            for i in range(len(self.sampled_pts)):
                lane_type = self.list_points_type[i]
                
                # Convert lane type to category using LANE_COLORS
                lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
                category = lane_info['category']
                
                # Prepare lane points
                # Convert points to row vectors
                xyz = self.sampled_pts[i].T.tolist()  # [x1, x2, x3...], [y1, y2, y3...], [z1, z2, z3...]
                uv = self.sampled_uv[i].T.tolist()    # [u1, u2, u3...], [v1, v2, v3...]
                
                # Create lane line data
                lane_line = {
                    "category": category,
                    "visibility": [1.0] * len(self.sampled_pts[i]),  # All points are visible
                    "uv": uv,
                    "xyz": xyz
                }
                
                lane_data.append(lane_line)
        
        # Create final data structure
        data = {
            "extrinsic": r.tolist(),  # Rotation matrix
            "intrinsic": k.tolist(),  # Intrinsic matrix
            "file_path": file_path,
            "lane_lines": lane_data
        }
        
        # Save to JSON file
        output_file = os.path.splitext(img_file)[0] + '.json'
        output_path = os.path.join(img_path, 'label', output_file)
        
        # Create label directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        try:
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved lane data to {output_path}")
        except Exception as e:
            print(f"Error saving file: {e}")