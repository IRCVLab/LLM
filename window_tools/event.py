from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QMessageBox


import os 
import vtk
import json
import numpy as np
import math
import open3d as o3d
import matplotlib.pyplot as plt

from utils.converter import load_calibration_params, projection_pcd_to_img, projection_img_to_pcd, interpolate_lane_curve, centripetal_catmull_rom

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
            # ----- drag state & event hook init -----
            if not hasattr(self, '_drag'):
                self._drag = {'active': False}
            if not hasattr(self, '_drag_hooks_set'):
                self._drag_hooks_set = False
            if not self._drag_hooks_set and hasattr(self, 'canvas'):
                # connect once
                self.canvas.mpl_connect('button_press_event',  self._on_mpl_press)
                self.canvas.mpl_connect('motion_notify_event', self._on_mpl_motion)
                self.canvas.mpl_connect('button_release_event', self._on_mpl_release)
                self._drag_hooks_set = True
            
    ##################################################################################
    ######################## Image View Event ########################################
    ##################################################################################

    def on_mpl_click(self, event):
        # Matplotlib canvas click event
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return

        # 드래그 처리 - 모든 점 검사 (끝점만 검사하던 문제 수정)
        thresh_px = 20
        for lane in getattr(self, 'unified_lanes', []):
            if not lane.get('points_2d'):
                continue
            # 모든 인덱스의 점 검사
            for ei, pt in enumerate(lane['points_2d']):
                if (event.xdata - pt[0])**2 + (event.ydata - pt[1])**2 <= thresh_px**2:
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
            print(f"[on_mpl_click] points_3d: {points_3d}")
            # --- 연선(폴리라인) 디버깅: 이미지→3D→VTK 변환 과정 ---
            if points_3d is not None:
                if points_3d.ndim == 1:
                    pts3d_iter = [points_3d]
                else:
                    pts3d_iter = points_3d
                for idx, pt3d in enumerate(pts3d_iter):
                    print(f"[DEBUG][IMG→3D] Polyline idx={idx} 3D point (lidar/world): {pt3d}")
                    pt = np.array(pt3d).reshape(3, 1)
                    pt_cam = r @ pt + t
                    print(f"[DEBUG][IMG→3D] Polyline idx={idx} pt_cam (camera coords): {pt_cam.ravel()} (z={pt_cam[2,0]})")
                print(f"[DEBUG][IMG→3D] Polyline 전체 {len(pts3d_iter)}개 3D 점 변환 완료.")
            # -------------------------------------------------------
            self.addLanePointsToVTK(points_3d, color=lane_info['vtk_color'], size=7, single_click=True)
            if not hasattr(self, 'vtk_lane_points'):
                self.vtk_lane_points = []
            self.vtk_lane_points.append(points_3d)

            self.is_lane_completed = False

        except Exception as e:
            print(f"[on_mpl_click] error: {e}")

        # ------------------------------------------------------------------
        # Drag handlers  (lane endpoints)
        # ------------------------------------------------------------------
    def _on_mpl_press(self, event):
        if event.button != 1 or event.xdata is None or event.ydata is None:
            return
        thresh_px = 10
        hit_li = hit_ei = None
        for li, lane in enumerate(getattr(self, 'unified_lanes', [])):
            if not lane.get('points_2d'):
                continue
            for ei, pt in enumerate(lane['points_2d']):
                dx, dy = event.xdata - pt[0], event.ydata - pt[1]
                if dx*dx + dy*dy <= thresh_px*thresh_px:
                    hit_li, hit_ei = li, ei
                    break
            if hit_li is not None:
                break
        if hit_li is None:
            return
        lane = self.unified_lanes[hit_li]
        self._drag.update(active=True,
                          lane_idx=hit_li,
                          end_idx=hit_ei,
                          mpl_artist=lane['img_points'][hit_ei] if hit_ei < len(lane.get('img_points',[])) else None,
                          vtk_actor=lane['vtk_point_actors'][hit_ei] if 'vtk_point_actors' in lane else None)

    def _on_mpl_motion(self, event):
        if not self._drag.get('active') or event.xdata is None or event.ydata is None:
            return
        li = self._drag['lane_idx']
        ei = self._drag['end_idx']
        lane = self.unified_lanes[li]
        # update 2d point
        lane['points_2d'][ei] = [event.xdata, event.ydata]
        
        # update the scatter point artist
        if 'img_points' in lane and ei < len(lane['img_points']) and lane['img_points'][ei] is not None:
            lane['img_points'][ei].set_offsets([event.xdata, event.ydata])
        # update the line artist with smooth curve
        if lane.get('img_curve') is not None:
            xs, ys = centripetal_catmull_rom(lane['points_2d'])
            lane['img_curve'].set_data(xs, ys)
        self.canvas.draw_idle()

    def _on_mpl_release(self, event):
        """Mouse release: commit the drag.
        – projects final 2D point to 3D
        – updates corresponding VTK polyline point (any index)
        – moves endpoint sphere actor if it exists (only for first/last)"""
        if not self._drag.get('active'):
            return
        li = self._drag['lane_idx']
        ei = self._drag['end_idx']
        lane = self.unified_lanes[li]
        x2d, y2d = lane['points_2d'][ei]

        # --- 2D → 3D projection ---
        try:
            # lazy load resources
            if not hasattr(self, 'pcd_points_np'):
                img_file = self.list_img_path[self.imgIndex]
                pcd_file = os.path.splitext(os.path.basename(img_file))[0] + '.pcd'
                pcd_path = os.path.join(self.pcd_dir, pcd_file)
                pcd = o3d.t.io.read_point_cloud(pcd_path)
                self.pcd_points_np = pcd.point.positions.numpy()
            if not hasattr(self, 'calibration_params'):
                t,r,k,_ = load_calibration_params()
            else:
                t,r,k,_ = self.calibration_params

            rgb_img = self.pyt.get_array() if hasattr(self, 'pyt') else None
            proj_res = projection_img_to_pcd(rgb_img, self.pcd_points_np, k, r, t,
                                             np.array([[x2d, y2d]]), single_click=True)
            pts3d = proj_res[0] if isinstance(proj_res, tuple) else proj_res
            pts3d = np.asarray(pts3d).reshape(-1)
            if pts3d.size < 3:
                raise ValueError('projection failed')
            nx, ny, nz = [float(v) for v in pts3d[:3]]
            lane['points_3d'][ei] = [nx, ny, nz]

            # --- update VTK polyline ---
            if lane.get('vtk_actor') is not None:
                polydata = lane['vtk_actor'].GetMapper().GetInput()
                vtk_pts = polydata.GetPoints()
                if vtk_pts is not None and ei < vtk_pts.GetNumberOfPoints():
                    vtk_pts.SetPoint(ei, nx, ny, nz)
                    vtk_pts.Modified()
            
                

            # --- update endpoint sphere actors (only if actor present for this idx) ---
            if lane.get('vtk_point_actors') and ei < len(lane['vtk_point_actors']):
                act = lane['vtk_point_actors'][ei]
                pd = act.GetMapper().GetInput(); pts = pd.GetPoints() if pd else None
                if pts is not None and pts.GetNumberOfPoints() > 0:
                    pts.SetPoint(0, nx, ny, nz)
                    pts.Modified(); act.GetMapper().Update(); act.Modified()
                act.SetPosition(0,0,0)
            lane['vtk_actor'].GetMapper().Update()
            
        except Exception as e:
            print('[release sync 2D→3D]', e)

        # update 2D curve with smooth interpolation
        if lane.get('img_curve') is not None:
            xs, ys = centripetal_catmull_rom(lane['points_2d'])
            lane['img_curve'].set_data(xs, ys)
            
        # final redraws
        self.canvas.draw_idle()
        if hasattr(self, 'vtkWidget'):
            self.vtkWidget.GetRenderWindow().Render()
        self._drag['active'] = False


    


    ##################################################################################
    ######################## VTK View Event ##########################################
    ##################################################################################

    def on_vtk_click(self, obj, event):
        # 클릭 위치 저장 (press)
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        
        if self._drag.get('active'):
            self._drag['active'] = False
            self.vtkWidget.GetRenderWindow().Render()
            return

        # otherwise: fresh click processing
        self._vtk_press_pos = interactor.GetEventPosition()
        self._vtk_point_added = False  # 실제 point 생성은 기존 코드에서 판단

        ctrl = interactor.GetControlKey()
        shift = interactor.GetShiftKey()
        alt = interactor.GetAltKey() if hasattr(interactor, "GetAltKey") else False
        
        # window control
        if ctrl or shift or alt:
            return
        x, y = interactor.GetEventPosition()
        print(f"[VTK CLICK] Display coordinates: x={x}, y={y}")
        renderer = self.vtkRenderer

        picker = vtk.vtkPointPicker()
        picker.Pick(x, y, 0, renderer)


        # ------------------------------------------------------------------
        # Try to find nearest existing point in SCREEN‐PIXEL space (<=10 px)
        # ------------------------------------------------------------------
        best_li = best_ei = None
        best_px2 = np.inf  # squared pixel distance
        debug_dists = []
        for li, lane in enumerate(getattr(self, 'unified_lanes', [])):
            # 반드시 point와 actor를 같이 zip으로 순회
            for ei, (p3, actor) in enumerate(zip(lane.get('points_3d', []), lane.get('vtk_point_actors', []))):
                if p3 is None:
                    continue
                renderer.SetWorldPoint(*p3, 1.0)
                renderer.WorldToDisplay()
                sx, sy, _ = renderer.GetDisplayPoint()
                dx = sx - x
                dy = sy - y
                d2 = dx*dx + dy*dy
                debug_dists.append((li, ei, math.sqrt(d2)))
                if d2 < best_px2:
                    best_px2 = d2
                    best_li, best_ei = li, ei
                    best_actor = actor
        tol_px = 10.0  # pixels
        tol2 = tol_px * tol_px
        if best_li is not None and best_px2 < tol2:
            interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
            lane = self.unified_lanes[best_li]
            # Derive display depth for correct drag mapping
            renderer.SetWorldPoint(*lane['points_3d'][best_ei], 1.0)
            renderer.WorldToDisplay()
            disp_z = renderer.GetDisplayPoint()[2]
            self._drag.update(active=True,
                              lane_idx=best_li,
                              end_idx=best_ei,
                              mpl_artist=lane['img_points'][best_ei] if best_ei < len(lane.get('img_points', [])) else None,
                              vtk_actor=best_actor,
                              vtk_depth=disp_z)
            print(f"[VTK click] Drag start on existing actor: lane={best_li}, idx={best_ei}")
            return  # handled as drag start

        print("[VTK click] Add new point.")

        if not (hasattr(self, 'pcd_points_np') and hasattr(self, 'pcd_colors_np')):
            try:
                pcd_file = self.list_pcd_path[self.imgIndex]
                pcd_path = os.path.join(self.pcd_dir, pcd_file)
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

        min_dist = float('inf')
        nearest = None
        renderer = self.vtkRenderer
        for pt in self.pcd_points_np:
            renderer.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
            renderer.WorldToDisplay()
            display_coords = renderer.GetDisplayPoint()
            dx = display_coords[0] - x
            dy = display_coords[1] - y
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                nearest = pt

        if nearest is not None:
            print(f"[VTK CLICK] Nearest world coordinates: {nearest}")
            renderer.SetWorldPoint(nearest[0], nearest[1], nearest[2], 1.0)
            renderer.WorldToDisplay()
            sx, sy, sz = renderer.GetDisplayPoint()

        # --- Accumulate clicked points for VTK polyline ---
        if not hasattr(self, 'vtk_lane_points'):
            self.vtk_lane_points = []
        self.vtk_lane_points.append(nearest)
        # VTK 점 시각화 (endpoint 클릭 시 해당 점 actor 제거)
        lane_type = None
        if self.beforeRadioChecked == self.lineRadio1:
            lane_type = 'Yellow'
        elif self.beforeRadioChecked == self.lineRadio2:
            lane_type = 'White'
        elif self.beforeRadioChecked == self.lineRadio3:
            lane_type = 'WhiteDash'
        lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
        vtk_color = lane_info['vtk_color']

        # --- endpoint 클릭 시 해당 점 actor 제거 ---
        # (unified_lanes의 endpoint와 일치하는지 검사)
        remove_idx = None
        for li, lane in enumerate(getattr(self, 'unified_lanes', [])):
            if not lane.get('points_3d'):
                continue
            # endpoint만 검사
            for ei in [0, len(lane['points_3d'])-1]:
                if np.allclose(lane['points_3d'][ei], nearest, atol=1e-4):
                    # VTK endpoint actor 제거
                    if 'vtk_point_actors' in lane and ei < len(lane['vtk_point_actors']):
                        actor = lane['vtk_point_actors'][ei]
                        self.vtkRenderer.RemoveActor(actor)
                        lane['vtk_point_actors'][ei] = None
                    remove_idx = (li, ei)
                    break
            if remove_idx:
                break
        # 점을 새로 추가하는 경우만 actor 생성
        self.addLanePointsToVTK(points=np.array([nearest]), color=vtk_color, size=7, single_click=True)
        # --- 3D→2D 변환 및 이미지에 시각화 ---
        try:
            if hasattr(self, "calibration_params"):
                t, r, k, _ = self.calibration_params
            else:
                t, r, k, _ = load_calibration_params()
                self.calibration_params = (t, r, k, _)
            img_shape = self.img.shape if hasattr(self, "img") else None

            # --- 추가: 카메라 좌표계 변환 결과 출력 ---
            pt = np.array(nearest).reshape(3, 1)
            pt_cam = r @ pt + t  # shape (3,1)
            # ------------------------------------------------
            points_2d = projection_pcd_to_img(
                np.array([nearest]), k, r, t, img_shape, single_click=True
            )
            # ---- validate projection within image bounds ----
            if points_2d is None or len(points_2d) == 0:
                self.delete_vtk_point()
                QMessageBox.warning(None, 'WARNING', 'Clicked point projects outside current image.')
                return
                
            x_img, y_img = float(points_2d[0][0]), float(points_2d[0][1])
            if img_shape is not None:
                h, w = img_shape[:2]
                margin = 10  # pixels margin from image edges
                # Check if point is outside image bounds or too close to edges
                if (x_img < margin or y_img < margin or 
                    x_img >= w - margin or y_img >= h - margin):
                    self.delete_vtk_point()
                    QMessageBox.warning(
                        None, 
                        'WARNING', 
                        f'Point is too close to image edge or outside. '
                        f'Keep points at least {margin}px from edges.\n'
                        f'Image size: {w}x{h}, Clicked: ({x_img:.1f}, {y_img:.1f})'
                    )
                    return
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
            self.is_lane_completed = False
        except Exception as e:
            print(f"[on_vtk_click] error: {e}")
        else:
            self._vtk_point_added = True  # 정상적으로 point가 생성된 경우만 True




    def on_vtk_release(self, obj, event):
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        release_pos = interactor.GetEventPosition()
        interactor.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())

        if self._drag.get('active'):
            self._drag['active'] = False
            return

        if self._vtk_point_added and hasattr(self, '_vtk_press_pos') and self._vtk_press_pos != release_pos:
            self.delete_vtk_point()
            self._vtk_point_added = False


    def on_vtk_motion(self, obj, event):
        if not self._drag.get('active') or self._drag.get('vtk_actor') is None:
            return
        interactor = self.vtkWidget.GetRenderWindow().GetInteractor()
        x, y = interactor.GetEventPosition()
        print(f"[VTK Motion] Display coordinates: x={x}, y={y}")
        renderer = self.vtkRenderer

        # 화면상 마우스 위치에서 가장 가까운 실제 포인트클라우드 점을 찾음
        min_dist = float('inf')
        nearest = None
        for pt in self.pcd_points_np:
            renderer.SetWorldPoint(pt[0], pt[1], pt[2], 1.0)
            renderer.WorldToDisplay()
            display_coords = renderer.GetDisplayPoint()
            dx = display_coords[0] - x
            dy = display_coords[1] - y
            dist = dx*dx + dy*dy
            if dist < min_dist:
                min_dist = dist
                nearest = pt
        if nearest is None:
            return
        nx, ny, nz = nearest[0], nearest[1], nearest[2]
        li = self._drag['lane_idx']
        ei = self._drag['end_idx']
        lane = self.unified_lanes[li]
        
        print(f"[VTK Motion] Mouse display: ({x:.1f}, {y:.1f})")
        print(f"[VTK Motion] Nearest world: [{nx:.3f}, {ny:.3f}, {nz:.3f}], min_dist={min_dist:.1f}")
        print(f"[VTK Motion] Dragging lane_idx={li}, end_idx={ei}")
        print(f"[VTK Motion] Old 3D: {lane['points_3d'][ei]} → New: [{nx:.3f}, {ny:.3f}, {nz:.3f}]")
        nearest_idx = np.where((self.pcd_points_np == nearest).all(axis=1))[0]
        if len(nearest_idx) > 0:
            print(f"[VTK Motion] Nearest point index in pcd: {nearest_idx[0]}")
        # 1. 데이터 갱신
        lane['points_3d'][ei] = [nx, ny, nz]
        # 2. 점 actor 이동 (드래그 중인 점은 actor를 항상 remove + None)
        if 'vtk_point_actors' in lane and ei < len(lane['vtk_point_actors']) and lane['vtk_point_actors'][ei] is not None:
            self.vtkRenderer.RemoveActor(lane['vtk_point_actors'][ei])
            lane['vtk_point_actors'][ei] = None
        # (actor 이동은 하지 않음: 항상 remove)
        # 3. 선 actor polydata 갱신 (곡선 보간 적용)
        if lane.get('vtk_actor') is not None:
            pts3d = np.array(lane['points_3d'])
            if len(pts3d) >= 3:
                curve_points = interpolate_lane_curve(pts3d, num_samples=70)
            else:
                curve_points = pts3d
            polydata = lane['vtk_actor'].GetMapper().GetInput()
            vtk_pts = polydata.GetPoints()
            # 점 개수가 다르면 새로 할당
            if vtk_pts is None or vtk_pts.GetNumberOfPoints() != len(curve_points):
                vtk_pts = vtk.vtkPoints()
                for pt in curve_points:
                    vtk_pts.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))
                polydata.SetPoints(vtk_pts)
            else:
                for i, pt in enumerate(curve_points):
                    vtk_pts.SetPoint(i, float(pt[0]), float(pt[1]), float(pt[2]))
                vtk_pts.Modified()
            lane['vtk_actor'].GetMapper().Update()
        # 4. 3D → 2D 프로젝션 및 이미지 아티스트 동기화
        try:
            if not hasattr(self, 'calibration_params'):
                t, r, k, _ = load_calibration_params()
            else:
                t, r, k, _ = self.calibration_params
            img_shape = self.img.shape if hasattr(self, 'img') else None
            pt2d = projection_pcd_to_img(np.array([[nx, ny, nz]]), k, r, t, img_shape, single_click=True)
            if pt2d is not None and len(pt2d) > 0:
                lane['points_2d'][ei] = [float(pt2d[0][0]), float(pt2d[0][1])]
                if 'img_points' in lane and ei < len(lane['img_points']) and lane['img_points'][ei] is not None:
                    lane['img_points'][ei].set_offsets(lane['points_2d'][ei])
                # Update curve
                if lane.get('img_curve') is not None:
                    xs, ys = centripetal_catmull_rom(lane['points_2d'])
                    lane['img_curve'].set_data(xs, ys)
                self.canvas.draw_idle()
        except Exception as e:
            print(f"[on_vtk_motion] projection error: {e}")
        # 5. 렌더 갱신
        self.vtkWidget.GetRenderWindow().Render()

        # --- 드래그 종료 후 점 actor 재생성 (모든 점에 대해) ---
        if 'vtk_point_actors' in lane:
            # 기존 actor를 None으로 비워뒀으니 새로 생성
            actor = self.addLanePointsToVTK(points=np.array([[nx, ny, nz]]), color=(1,0,0), size=7, single_click=True, return_actor=True)
            lane['vtk_point_actors'][ei] = actor

    def vtkViewClicked(self):
        # --- 기존 point cloud actor만 제거 (lane 등은 유지) ---
        if hasattr(self, 'pcd_vtk_actor') and self.pcd_vtk_actor is not None:
            try:
                self.vtkRenderer.RemoveActor(self.pcd_vtk_actor)
            except Exception:
                pass
            self.pcd_vtk_actor = None
        # --- 새 point cloud 추가 ---
        if self.colorRadio.isChecked():
            actor = self.addColoredPointCloudToVTK(self.imgIndex, return_actor=True)
            self.pcd_vtk_actor = actor
        elif self.intensityRadio.isChecked():
            actor = self.addPointCloudToVTK(self.imgIndex, return_actor=True)
            self.pcd_vtk_actor = actor
        
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
        if self.is_lane_completed:
            QMessageBox.warning(None, 'WARNING','Cannot delete point of completed lane')
            return
        # 이미지 점 삭제
        self.delete_img_point()
        # VTK 점 삭제
        self.delete_vtk_point()
        
    def delete_img_point(self):
        if hasattr(self, 'lane_points') and self.lane_points:
            self.lane_points.pop()
        if hasattr(self, 'lane_point_artists') and self.lane_point_artists:
            last_artist = self.lane_point_artists.pop()
            last_artist.remove()
        if hasattr(self, 'canvas'):
            self.canvas.draw()
        


    def delete_vtk_point(self):
        # if hasattr(self, 'vtk_lanes') and self.vtk_lanes:
        #     last_lane = self.vtk_lanes[-1]
        #     # numpy 배열인지, 파이썬 리스트인지 구분
        #     if isinstance(last_lane, np.ndarray):
        #         if last_lane.shape[0] > 0:
        #             # 배열에서 마지막 행 제거
        #             self.vtk_lanes[-1] = last_lane[:-1]
        #         # 비어 있으면 레인 자체 제거
        #         if self.vtk_lanes[-1].shape[0] == 0:
        #             self.vtk_lanes.pop()
        #     elif isinstance(last_lane, list):
        #         if last_lane:
        #             last_lane.pop()
        #         if not last_lane:
        #             self.vtk_lanes.pop()


        if hasattr(self, 'vtk_lane_points') and self.vtk_lane_points:
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
            # img_points_aligned = [None] * len(points_2d)
            # for idx, art in enumerate(img_click_actors):
            #     if idx < len(img_points_aligned):
            #         img_points_aligned[idx] = art

            # vtk_point_actors_aligned = [None] * len(points_2d)
            # for idx, act in enumerate(vtk_click_actors):
            #     if idx < len(vtk_point_actors_aligned):
                    # vtk_point_actors_aligned[idx] = act
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
            for a in img_click_actors:
                if a not in lane_data['img_points']:
                    lane_data['img_points'].append(a)
        if 'vtk_point_actors' in lane_data:
            for act in vtk_click_actors:
                if act not in lane_data['vtk_point_actors']:
                    lane_data['vtk_point_actors'].append(act)

        # Store
        self.unified_lanes.append(lane_data)

        self.is_lane_completed = True

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
        # Save label JSON in ./data/label/<image_basename>.json
        img_file = os.path.basename(self.list_img_path[ind])       # e.g. 000022.jpg
        # Relative paths for dataset structure
        image_rel_path = os.path.join("data", "image", img_file)   # e.g. data/image/000022.jpg
        label_dir = os.path.join("data", "label")                  # ./data/label
        os.makedirs(label_dir, exist_ok=True)
        json_name = os.path.splitext(img_file)[0] + ".json"
        label_json_path = os.path.join(label_dir, json_name)
        
        # Prepare lane data
        lane_data = []
        
        # 통합 레인 데이터 구조에서 저장
        if hasattr(self, 'unified_lanes') and self.unified_lanes:
            num_samples = 20  # 등간격 샘플링 개수


            for idx, lane in enumerate(self.unified_lanes):
                lane_type = lane.get('type', 'Default')
                lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
                category = lane_info['category']

                pts3d = None
                # 1순위: points_3d 사용
                if 'points_3d' in lane and lane['points_3d'] is not None:
                    pts3d = np.array(lane['points_3d'])
                # 2순위: VTK actor에서 추출
                elif 'vtk_actor' in lane and lane['vtk_actor'] is not None:
                    actor = lane['vtk_actor']
                    try:
                        polydata = actor.GetMapper().GetInput()
                        vtk_points = polydata.GetPoints()
                        num_points = vtk_points.GetNumberOfPoints()
                        pts3d = np.array([vtk_points.GetPoint(i) for i in range(num_points)])  # (N, 3)
                        print(f"[saveAll] pts3d from vtk_actor: {pts3d}")
                    except Exception as e:
                        print(f"[saveAll] VTK actor 3D extraction error: {e}")
                # 3순위: fallback - None

                sampled_3d = interpolate_lane_curve(pts3d, num_samples=num_samples)
                sampled_2d = projection_pcd_to_img(sampled_3d, k, r, t)


                if sampled_2d.shape[0] < 2:
                    continue

                xyz = sampled_3d.T.tolist()  # (num_samples,3)→(3,N)
                uv = sampled_2d.tolist()
                lane_line = {
                    "category": category,
                    "visibility": [1.0] * len(xyz[0]),
                    "uv": uv,
                    "xyz": xyz,
                    "track_id": idx + 1,
                    "attribute": 0
                }
                lane_data.append(lane_line)

        data = {
            "extrinsic": r.tolist(),  # Rotation matrix
            "intrinsic": k.tolist(),  # Intrinsic matrix
            "file_path": image_rel_path,
            "lane_lines": lane_data
        }

        try:
            with open(label_json_path, 'w') as f:
                json.dump(data, f, indent=4)
            print(f"Saved lane data to {label_json_path}")
        except Exception as e:
            print(f"Error saving file: {e}")