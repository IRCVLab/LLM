import os
import vtk
import numpy as np
import open3d as o3d

from scipy.interpolate import interp1d

from utils.polynomial import centripetal_catmull_rom

class VizTools:

    def addPointCloudToVTK(self, index=None):
        if index is None:
            index = self.imgIndex if self.imgIndex >= 0 else 0
        img_file = self.list_img_path[index]
        pcd_file = os.path.splitext(os.path.basename(img_file))[0] + '.pcd'
        pcd_path = os.path.join(self.pcd_dir, pcd_file)
        if not os.path.exists(pcd_path):
            print(f"PCD 파일이 존재하지 않습니다: {pcd_path}")
            return
        pcd = o3d.t.io.read_point_cloud(pcd_path)
        np_points = pcd.point.positions.numpy()
        
        vtk_points = vtk.vtkPoints()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        
        # 색상이 있는 경우와 없는 경우 분기 처리
        if "colors" in pcd.point:
            np_colors = pcd.point.colors.numpy()
            for pt, rgb in zip(np_points, np_colors):
                vtk_points.InsertNextPoint(*pt)
                vtk_colors.InsertNextTuple3(*rgb)
        else:
            # 색상이 없으면 회색으로 설정
            gray_color = [128, 128, 128]
            for pt in np_points:
                vtk_points.InsertNextPoint(*pt)
                vtk_colors.InsertNextTuple3(*gray_color)

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.GetPointData().SetScalars(vtk_colors)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vertex_filter.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)

        self.vtkRenderer.AddActor(actor)
        self.vtkRenderer.ResetCamera()
        
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.SetMotionFactor(4)
        self.vtkWidget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)

        self.vtkRenderer.GetActiveCamera().Zoom(1.2)
        
        self.vtkWidget.GetRenderWindow().Render()

    def addLanePolylineToVTK(self, points, color=(1,0,0), width=4):

        if points is None or len(points) < 2:
            return

        if not hasattr(self, 'lane_vtk_polyline_actors'):
            self.lane_vtk_polyline_actors = []

        if len(points) >= 3:
            num_samples = 70
            xs, ys = centripetal_catmull_rom(points[:, :2], num_samples=num_samples)

            pts = np.asarray(points, dtype=float)
            deltas = np.diff(pts, axis=0)
            dist_alpha = np.sqrt((deltas**2).sum(axis=1))**0.5
            t_orig = [0.0]
            for da in dist_alpha:
                t_orig.append(t_orig[-1] + da)
            t_orig = np.array(t_orig)
            t_orig /= t_orig[-1]
            t_lin = np.linspace(0, 1, num_samples)

            t_orig_unique, unique_indices = np.unique(t_orig, return_index=True)
            z_unique = points[:,2][unique_indices]
            interp_kind = 'cubic' if len(t_orig_unique) > 3 else 'linear'
            interp_z = interp1d(t_orig_unique, z_unique, kind=interp_kind)
            zs = interp_z(t_lin)
            curve_points = np.stack([xs, ys, zs], axis=1)

        else:
            curve_points = points
        vtk_points = vtk.vtkPoints()
        for pt in curve_points:
            vtk_points.InsertNextPoint(*pt)
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(curve_points))
        for i in range(len(curve_points)):
            lines.InsertCellPoint(i)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        polydata.SetLines(lines)
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        rgb = [float(c) for c in color]
        actor.GetProperty().SetColor(*rgb)
        actor.GetProperty().SetLineWidth(width)
        self.vtkRenderer.AddActor(actor)
        self.lane_vtk_polyline_actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()


    def addLanePointsToVTK(self, points, color=(1,0,0), size=10, single_click=False):
        """
        points: (N, 3) numpy array
        color: RGB tuple (0~1)
        size: point size
        """

        if points is None or len(points) == 0:
            print("lane_points_3d가 비어 있습니다.")
            return

        # Build vtkPoints from input array/list
        vtk_points = vtk.vtkPoints()
        arr = np.asarray(points)
        if single_click:
            arr = arr.reshape(1, 3)
        if arr.ndim == 1:
            arr = arr.reshape(1, 3)
        for pt in arr:
            vtk_points.InsertNextPoint(float(pt[0]), float(pt[1]), float(pt[2]))

        # Polydata wrapper
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        # Create 2D circle glyph as screen-space disc
        glyph_source = vtk.vtkGlyphSource2D()
        glyph_source.SetGlyphTypeToCircle()
        glyph_source.FilledOn()
        glyph_source.SetResolution(20)

        glyph3d = vtk.vtkGlyph3D()
        glyph3d.SetSourceConnection(glyph_source.GetOutputPort())
        glyph3d.SetInputData(polydata)
        # Use a uniform scale factor that affects glyph geometry only (does not translate center)
        sf = max(size, 2) * 0.08  # heuristic pixel->world conversion
        glyph3d.SetScaleFactor(sf)
        glyph3d.ScalingOn()
        glyph3d.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(glyph3d.GetOutputPort())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Interior white fill
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        # Colored outline
        rgb = [float(c) for c in color]
        actor.GetProperty().SetEdgeVisibility(1)
        actor.GetProperty().SetEdgeColor(*rgb)
        actor.GetProperty().SetLineWidth(2)

        self.vtkRenderer.AddActor(actor)
        self.vtkWidget.GetRenderWindow().Render()

        if not hasattr(self, 'lane_vtk_actors'):
            self.lane_vtk_actors = []
        self.lane_vtk_actors.append(actor)


    def draw_lane_from_unified(self, lane_data):
        """
        통합 레인 데이터로부터 2D 및 3D 시각화를 수행하는 함수
        
        Args:
            lane_data: 통합 레인 데이터 딕셔너리
                {
                    'type': 레인 타입 (Yellow, White, WhiteDash),
                    'points_2d': 2D 좌표 리스트,
                    'points_3d': 3D 좌표 리스트,
                    'img_points': 이미지 점 아티스트 리스트,
                    'vtk_points': VTK 점 좌표 리스트
                }
        """
        # 레인 타입에 따른 색상 정보 가져오기
        lane_type = lane_data.get('type')
        lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
        color = lane_info['mpl_color']
        vtk_color = lane_info['vtk_color']
        linestyle = '-' if lane_type != 'WhiteDash' else '--'
        
        # 1. 이미지 뷰에 레인 그리기 (2D 좌표 사용)
        points_2d = lane_data.get('points_2d', [])
        if points_2d and len(points_2d) >= 2:
            # 곡선 그리기
            xs, ys = centripetal_catmull_rom(points_2d)
            curve_artist, = self.axes.plot(xs, ys, linestyle, color=color, linewidth=2)
            lane_data['img_curve'] = curve_artist
            if hasattr(self, 'lane_curve_artists'):
                self.lane_curve_artists.append(curve_artist)
            
            # 점 아티스트 생성 (없는 경우)
            if not lane_data.get('img_points'):
                img_points = []
                for x, y in points_2d:
                    point_artist = self.axes.scatter(x, y, color=color, s=30, zorder=5)
                    img_points.append(point_artist)
                    if hasattr(self, 'all_point_artists'):
                        self.all_point_artists.append(point_artist)
                lane_data['img_points'] = img_points
            
            self.canvas.draw()
        
        # 2. VTK 뷰에 레인 그리기 (3D 좌표 사용)
        points_3d = lane_data.get('points_3d', [])
        vtk_points = lane_data.get('vtk_points', [])
        
        # 3D 좌표가 있으면 VTK에 그리기
        if points_3d and len(points_3d) >= 2:
            # numpy 배열로 변환
            points_array = np.array(points_3d)
            
            # ----------- create two endpoint actors (start, end) -----------
            prev_len = len(self.lane_vtk_actors) if hasattr(self,'lane_vtk_actors') else 0
            for idx in (0, -1):
                self.addLanePointsToVTK(points_array[idx], color=vtk_color, size=7, single_click=True)
            if hasattr(self,'lane_vtk_actors'):
                lane_data['vtk_point_actors'] = self.lane_vtk_actors[prev_len:]
            else:
                lane_data['vtk_point_actors'] = []
            # ----------- VTK polyline 추가
            self.addLanePolylineToVTK(points=points_array, color=vtk_color, width=3)
            if hasattr(self, 'lane_vtk_polyline_actors') and self.lane_vtk_polyline_actors:
                lane_data['vtk_actor'] = self.lane_vtk_polyline_actors[-1]
            
            # vtk_lanes에 추가 (삭제 시 필요)
            if hasattr(self, 'vtk_lanes'):
                self.vtk_lanes.append(points_array)
        
        return lane_data

    # def draw_lane_i2v(self, ):
    #     if len(self.lane_points) < 2:
    #         return
    #     xs, ys = centripetal_catmull_rom(self.lane_points)
    #     lane_type = None
    #     if self.beforeRadioChecked == self.lineRadio1:  # Yellow Line
    #         lane_type = 'Yellow'
    #     elif self.beforeRadioChecked == self.lineRadio2:  # White Line
    #         lane_type = 'White'
    #     elif self.beforeRadioChecked == self.lineRadio3:  # White Dash Line
    #         lane_type = 'WhiteDash'
    #     if lane_type:
    #         lane_info = self.LANE_COLORS[lane_type]
    #         color = lane_info['mpl_color']
    #         vtk_color = lane_info['vtk_color']
    #         linestyle = '-' if lane_type != 'WhiteDash' else '--'
    #         self.list_points_type.append(lane_type)
    #     else:
    #         lane_info = self.LANE_COLORS['Default']
    #         color = lane_info['mpl_color']
    #         vtk_color = lane_info['vtk_color']
    #         linestyle = '-'
    #         self.list_points_type.append('Unknown')
    #     curve_artist, = self.axes.plot(xs, ys, linestyle, color=color)
    #     self.lane_curve_artists.append(curve_artist)
    #     n = len(self.lane_points)
    #     if self.lane_labels:
    #         last_end = self.lane_labels[-1][1]
    #     else:
    #         last_end = 0
    #     self.lane_labels.append((last_end, last_end + n))
    #     used_points = self.lane_points.copy()
    #     self.lane_points = []
    #     self.lane_point_artists = []
    #     self.canvas.draw()
    #     lane_polyline_img_coords = [(int(x), int(y)) for x, y in zip(xs, ys)]
    #     rgb_image = self.pyt.get_array() if hasattr(self, 'pyt') else None
    #     if rgb_image is None:
    #         print("Image data not available")
    #         return
        
    #     img_file = self.list_img_path[self.imgIndex]
    #     pcd_file = os.path.splitext(os.path.basename(img_file))[0] + '.pcd'
    #     pcd_path = os.path.join(self.pcd_dir, pcd_file)
    #     if not os.path.exists(pcd_path):
    #         print(f"PCD file not found: {pcd_path}")
    #         return
    #     pcd = o3d.t.io.read_point_cloud(pcd_path)
    #     np_points = pcd.point.positions.numpy()
    #     lane_points_3d, points_2d = projection_img_to_pcd(
    #         rgb_image,
    #         np_points,
    #         self.k,
    #         self.r,
    #         self.t,
    #         lane_polyline_img_coords,
    #         lane_thickness=3
    #     )
    #     if lane_points_3d.shape[0] == 4:
    #         points_3d = lane_points_3d[:3, :].T
    #     else:
    #         points_3d = lane_points_3d.T
    #     new_sampled_pts, new_sampled_uv = sample_lane_points(points_3d.T, points_2d, num_samples=20)
    #     self.sampled_pts.append(new_sampled_pts)
    #     self.sampled_uv.append(new_sampled_uv)
    #     self.addLanePointsToVTK(new_sampled_pts, color=vtk_color, size=5)


    # def draw_lane_v2i(self):
    #     # 1. 이미지(plt)에서 찍은 점으로 기존대로 처리
    #     if len(self.lane_points) >= 2:
    #         self.draw_lane_curve()
    #     # 2. VTK에서 찍은 점이 2개 이상이면 VTK에 선 그리기 (여러 라인 지원)
    #     if not hasattr(self, 'vtk_lanes'):
    #         self.vtk_lanes = []
    #     if hasattr(self, 'vtk_lane_points') and len(self.vtk_lane_points) >= 2:
    #         lane_type = None
    #         if self.beforeRadioChecked == self.lineRadio1:
    #             lane_type = 'Yellow'
    #         elif self.beforeRadioChecked == self.lineRadio2:
    #             lane_type = 'White'
    #         elif self.beforeRadioChecked == self.lineRadio3:
    #             lane_type = 'WhiteDash'
    #         lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
    #         color = lane_info['mpl_color']
    #         vtk_color = lane_info['vtk_color']
    #         linestyle = '-' if lane_type != 'WhiteDash' else '--'
    #         # VTK polyline → 이미지로 투영
            
    #         rgb_image = self.pyt.get_array() if hasattr(self, 'pyt') else None
    #         img_shape = rgb_image.shape if rgb_image is not None else None
    #         lane_polyline_lidar = np.array(self.vtk_lane_points)  # (N, 3)
    #         lane_polyline_img = projection_pcd_to_img(
    #             lane_polyline_lidar,
    #             self.k,
    #             self.r,
    #             self.t,
    #             img_shape
    #         )  # (M, 2)
    #         # 이미지 위에 2D polyline 시각화 (matplotlib)

    #         if hasattr(self, 'axes') and lane_polyline_img.shape[0] >= 2 and img_shape is not None:
    #             import cv2
    #             H, W = img_shape[:2]
    #             clipped_segments = []
    #             for i in range(len(lane_polyline_img)-1):
    #                 pt1 = tuple(np.round(lane_polyline_img[i]).astype(int))
    #                 pt2 = tuple(np.round(lane_polyline_img[i+1]).astype(int))
    #                 inside, p1, p2 = cv2.clipLine((0,0,W-1,H-1), pt1, pt2)
    #                 if inside:
    #                     xs = [p1[0], p2[0]]
    #                     ys = [p1[1], p2[1]]
    #                     curve_artist, = self.axes.plot(xs, ys, linestyle, color=color, linewidth=2, zorder=10)
    #                     self.lane_curve_artists.append(curve_artist)
    #             self.canvas.draw()
    #         else:
    #             print("[DEBUG] Cannot draw: axes?", hasattr(self, 'axes'), "polyline shape:", lane_polyline_img.shape)

    #         # 기존대로 VTK에도 선 추가
    #         self.vtk_lanes.append(self.vtk_lane_points.copy())
    #         self.addLanePolylineToVTK(points=lane_polyline_lidar, color=vtk_color, width=3)
    #         self.vtk_lane_points = []  # 버퍼 초기화