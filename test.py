from PyQt5 import QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import numpy as np
import os
import open3d as o3d

class VTKViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(VTKViewer, self).__init__(parent)
        self.frame = QtWidgets.QFrame()
        self.vl = QtWidgets.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.renderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.renderer)

        self.load_pointcloud()

        self.renderer.ResetCamera()
        self.show()
        self.vtkWidget.Initialize()

    def load_pointcloud(self):
        # Example: generate random points
        pcd_path = os.path.join('000000.pcd')
        points = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(points.points)
        
        vtk_points = vtk.vtkPoints()
        for pt in points:
            vtk_points.InsertNextPoint(pt.tolist())

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vertex_filter.GetOutput())

        actor = vtk.vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(2)

        self.renderer.AddActor(actor)

if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    window = VTKViewer()
    window.setWindowTitle("VTK PointCloud Viewer")
    window.resize(800, 600)
    window.show()
    app.exec_()
