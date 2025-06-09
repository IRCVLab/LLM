import os
import numpy as np
import open3d as o3d
from PyQt5.QtWidgets import QMessageBox
from polynomial import centripetal_catmull_rom
from converter import load_calibration_params, lane_points_3d_from_pcd_and_lane, lidar_points_in_image, sample_lane_points
import vtk

class LaneTools:

    def addLanePointsToVTK(self, points, color=(1,0,0), size=10):
        """
        points: (N, 3) numpy array
        color: RGB tuple (0~1)
        size: point size
        """
        if points is None or len(points) == 0:
            print("lane_points_3d가 비어 있습니다.")
            return
        vtk_points = vtk.vtkPoints()
        for pt in points:
            vtk_points.InsertNextPoint(*pt)
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        rgb = [int(c*255) for c in color]
        for _ in range(points.shape[0]):
            vtk_colors.InsertNextTuple3(*rgb)
        polydata.GetPointData().SetScalars(vtk_colors)
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vertex_filter.GetOutput())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(size)
        self.vtkRenderer.AddActor(actor)
        self.vtkWidget.GetRenderWindow().Render()
        if not hasattr(self, 'lane_vtk_actors'):
            self.lane_vtk_actors = []
        self.lane_vtk_actors.append(actor)
        self.vtkWidget.GetRenderWindow().Render()

    def draw_lane_curve(self):
        if len(self.lane_points) < 2:
            return
        xs, ys = centripetal_catmull_rom(self.lane_points)
        lane_type = None
        if self.beforeRadioChecked == self.lineRadio1:  # Yellow Line
            lane_type = 'Yellow'
        elif self.beforeRadioChecked == self.lineRadio2:  # White Line
            lane_type = 'White'
        elif self.beforeRadioChecked == self.lineRadio3:  # White Dash Line
            lane_type = 'WhiteDash'
        if lane_type:
            lane_info = self.LANE_COLORS[lane_type]
            color = lane_info['mpl_color']
            vtk_color = lane_info['vtk_color']
            linestyle = '-' if lane_type != 'WhiteDash' else '--'
            self.list_points_type.append(lane_type)
        else:
            lane_info = self.LANE_COLORS['Default']
            color = lane_info['mpl_color']
            vtk_color = lane_info['vtk_color']
            linestyle = '-'
            self.list_points_type.append('Unknown')
        curve_artist, = self.axes.plot(xs, ys, linestyle, color=color)
        self.lane_curve_artists.append(curve_artist)
        n = len(self.lane_points)
        if self.lane_labels:
            last_end = self.lane_labels[-1][1]
        else:
            last_end = 0
        self.lane_labels.append((last_end, last_end + n))
        used_points = self.lane_points.copy()
        self.lane_points = []
        self.lane_point_artists = []
        self.canvas.draw()
        lane_polyline_img_coords = [(int(x), int(y)) for x, y in zip(xs, ys)]
        rgb_image = self.pyt.get_array() if hasattr(self, 'pyt') else None
        if rgb_image is None:
            print("이미지 배열을 찾을 수 없습니다.")
            return
        img_file = self.list_img_path[self.imgIndex]
        pcd_file = os.path.splitext(os.path.basename(img_file))[0] + '.pcd'
        pcd_path = os.path.join(self.pcd_dir, pcd_file)
        if not os.path.exists(pcd_path):
            print(f"PCD 파일이 존재하지 않습니다: {pcd_path}")
            return
        pcd = o3d.t.io.read_point_cloud(pcd_path)
        np_points = pcd.point.positions.numpy()
        lane_points_3d, points_2d = lane_points_3d_from_pcd_and_lane(
            rgb_image,
            np_points,
            self.k,
            self.r,
            self.t,
            lane_polyline_img_coords,
            lane_thickness=3
        )
        if lane_points_3d.shape[0] == 4:
            points_3d = lane_points_3d[:3, :].T
        else:
            points_3d = lane_points_3d.T
        new_sampled_pts, new_sampled_uv = sample_lane_points(points_3d.T, points_2d, num_samples=20)
        print(f"새로운 sampled point 개수: {len(new_sampled_pts)}")
        self.sampled_pts.append(new_sampled_pts)
        self.sampled_uv.append(new_sampled_uv)
        all_points = np.vstack(self.sampled_pts)
        all_uv = np.vstack(self.sampled_uv)
        print(f"누적된 sampled point 총 개수: {len(all_points)}")
        self.addLanePointsToVTK(new_sampled_pts, color=vtk_color, size=5)
