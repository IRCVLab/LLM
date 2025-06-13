import os
import open3d as o3d
import numpy as np
import vtk

class VTKTools:
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