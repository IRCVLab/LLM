#!/usr/bin/env python
# -*- coding:utf8 -*-
import os, sys
import matplotlib
matplotlib.use("Qt5Agg")
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPen, QColor, QPainterPath, QPixmap, QBrush


from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.path import Path
import matplotlib.patches as patches
import matplotlib.image as mpimg
import numpy as np
import cv2
import random
#from threading import Timer

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtk
import open3d as o3d
import matplotlib.cm as cm
import matplotlib.colors as colors

# Personnal modules
from marker import DraggablePoint
from updater import DelayedUpdater
from mouse_event import QtMouseEventFilter
from polynomial import centripetal_catmull_rom
from calibration import load_calibration_params, lane_points_3d_from_pcd_and_lane, lidar_points_in_image

# set initial 4 points
x1=800
y1=100

x2=500
y2=200

x3=300
y3=300

x4=200
y4=500

class Window(QWidget):
    imgIndex = -1
    saveFlag = True

    def __init__(self,path):
        super(Window,self).__init__()
        self.img_path = os.getcwd() + os.sep + path  # path는 'data'
        self.img_dir = os.path.join(self.img_path, 'image')
        self.pcd_dir = os.path.join(self.img_path, 'pcd')

        # a figure instance to plot on
        self.figure = Figure(tight_layout=True, dpi=96)
        self.axes = self.figure.add_subplot(111)
        self.axes.set_axis_off()
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)

        # this is the Canvas Widget that displays the `figure`
        # it takes the `figure` instance as a parameter to __init__
        self.canvas = FigureCanvas(self.figure)
        sizePolicy = QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setSizePolicy(sizePolicy)
        self.canvas.updateGeometry()

        # To store the draggable polygon
        self.list_points = []
        self.list_points_type = []

        # calibration params
        self.t, self.r, self.k, self.distortion = load_calibration_params()

        # To store img path
        self.list_img_path = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith('.jpg') or f.endswith('.png')
        ])

        # PCD 파일명 리스트
        self.list_pcd_path = sorted([
            f for f in os.listdir(self.pcd_dir)
            if f.endswith('.pcd')
        ])

        self.loadImg(self.img_path)


        # LiDAR widget
        # self.vtk_actor = vtk.vtkActor()
        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vtkRenderer = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.vtkRenderer)
        self.addPointCloudToVTK()


        self.plotBackGround(self.img_path,0,True)

        addLaneLabel = QLabel()
        addLaneLabel.setText("Label Setting")

        addLaneListButton = QComboBox()
        addLaneListButton.addItems(["---Select line type---","White line", "White dash line", "Yellow line"])
        addLaneListButton.activated.connect(self.addNewLine)

        lineGroupBox = QGroupBox("Line", self)
        # polygonGroupBox = QGroupBox("Polygon", self)
        # objectGroupBox = QGroupBox("Object", self)
        # pointGroupBox = QGroupBox("Point", self)

        self.lineRadio1 = QRadioButton("Yellow Line", self)
        self.lineRadio1.setChecked(True)
        self.beforeRadioChecked = self.lineRadio1
        self.lineRadio1.clicked.connect(self.radioButtonClicked)

        self.lineRadio2 = QRadioButton("White Line", self)
        self.lineRadio2.clicked.connect(self.radioButtonClicked)

        self.lineRadio3 = QRadioButton("White Dash Line", self)
        self.lineRadio3.clicked.connect(self.radioButtonClicked)

        # self.polygonRadio1 = QRadioButton("Crosswalk", self)
        # self.polygonRadio1.clicked.connect(self.radioButtonClicked)

        # self.polygonRadio2 = QRadioButton("Stop Line", self)
        # self.polygonRadio2.clicked.connect(self.radioButtonClicked)

        # self.polygonRadio3 = QRadioButton("Speed Dump", self)
        # self.polygonRadio3.clicked.connect(self.radioButtonClicked)

        # self.objectRadio1 = QRadioButton("Arrow", self)
        # self.objectRadio1.clicked.connect(self.radioButtonClicked)

        # self.objectRadio2 = QRadioButton("Diamond", self)
        # self.objectRadio2.clicked.connect(self.radioButtonClicked)

        # self.objectRadio3 = QRadioButton("Road Sign", self)
        # self.objectRadio3.clicked.connect(self.radioButtonClicked)

        # self.pointRadio1 = QRadioButton("Vanishing Point", self)
        # self.pointRadio1.clicked.connect(self.radioButtonClicked)

        addLaneButton = QPushButton("Add Label")
        addLaneButton.clicked.connect(self.draw_lane_curve)

        delLaneButton = QPushButton("Delete Last Label")
        delLaneButton.clicked.connect(self.delete_last_label)

        # addPointButton = QPushButton("Add Point")
        # addPointButton.clicked.connect(self.addNewPoint)

        delPointButton = QPushButton("Del Point")
        delPointButton.clicked.connect(self.delete_last_point)

        # loadPLButton = QPushButton("Load Previous Labels")
        # loadPLButton.clicked.connect(self.loadPrevLabel)

        curPosButton = QPushButton("Show current Labels")
        curPosButton.clicked.connect(self.showPosition)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        nextImgButton = QPushButton("Next Image")
        #nextImgButton.clicked.connect(lambda: self.plotBackGround(self.img_path,0))
        nextImgButton.clicked.connect(self.loadNextImage)

        preImgButton = QPushButton("Prev Image")
        #preImgButton.clicked.connect(lambda: self.plotBackGround(self.img_path,1))
        preImgButton.clicked.connect(self.loadPrevImage)

        saveButton = QPushButton("Save")
        saveButton.clicked.connect(lambda: self.saveAll(self.img_path))

        self.editBox = QPlainTextEdit()
        self.editBox.setFixedWidth(160)
        self.editBox.setPlainText("")
        self.editBox.setReadOnly(True)
        self.editBox.setDisabled(True)


        # Prepare a group button

        preNextLayout = QHBoxLayout()
        preNextLayout.addWidget(preImgButton)
        preNextLayout.addWidget(nextImgButton)

        saveLayout = QVBoxLayout()
        saveLayout.addLayout(preNextLayout)
        saveLayout.addWidget(saveButton)

        lineGrouplayout = QVBoxLayout()
        lineGroupBox.setLayout(lineGrouplayout)
        lineGrouplayout.addWidget(self.lineRadio1)
        lineGrouplayout.addWidget(self.lineRadio2)
        lineGrouplayout.addWidget(self.lineRadio3)

        # polygonGrouplayout = QVBoxLayout()
        # polygonGroupBox.setLayout(polygonGrouplayout)
        # polygonGrouplayout.addWidget(self.polygonRadio1)
        # polygonGrouplayout.addWidget(self.polygonRadio2)
        # polygonGrouplayout.addWidget(self.polygonRadio3)

        # objectGrouplayout = QVBoxLayout()
        # objectGroupBox.setLayout(objectGrouplayout)
        # objectGrouplayout.addWidget(self.objectRadio1)
        # objectGrouplayout.addWidget(self.objectRadio2)
        # objectGrouplayout.addWidget(self.objectRadio3)

        # pointGrouplayout = QVBoxLayout()
        # pointGroupBox.setLayout(pointGrouplayout)
        # pointGrouplayout.addWidget(self.pointRadio1)

        addDelLayout = QHBoxLayout()
        # addDelLayout.addWidget(addPointButton)
        addDelLayout.addWidget(delPointButton)

        addLayout = QVBoxLayout()
        addLayout.addWidget(addLaneLabel)
        #addLayout.addWidget(addLaneListButton)

        addLayout.addWidget(lineGroupBox)
        # addLayout.addWidget(polygonGroupBox)
        # addLayout.addWidget(objectGroupBox)
        # addLayout.addWidget(pointGroupBox)

        addLayout.addWidget(addLaneButton)
        addLayout.addWidget(delLaneButton)
        addLayout.addLayout(addDelLayout)

        rightLayout = QVBoxLayout()
        rightLayout.addLayout(addLayout)
        rightLayout.addSpacing(20)
        # rightLayout.addWidget(loadPLButton)
        rightLayout.addSpacing(20)
        rightLayout.addWidget(curPosButton)
        rightLayout.addWidget(self.editBox)
        rightLayout.addSpacing(20)
        rightLayout.addSpacerItem(verticalSpacer)
        rightLayout.addLayout(saveLayout)
        
        

        pcdWidget = QWidget()
        pcdWidget.setLayout(QVBoxLayout())
        pcdWidget.layout().addWidget(self.vtkWidget)
        
        layout = QHBoxLayout()
        layout.addWidget(self.canvas, 3)
        layout.addWidget(pcdWidget, 2)
        layout.addLayout(rightLayout)
        self.setLayout(layout)

        # Prevent "QWidget::repaint: Recursive repaint detected"
        self.delayer = DelayedUpdater(self.canvas)

        # for first label load bug
        #t = Timer(1.0, self.loadFirstLabel)
        #t.start()


        """
        upperLayout = QVBoxLayout()
        upperLayout.addWidget(nextImgButton)
        upperLayout.addWidget(preImgButton)
        upperLayout.addWidget(curPosButton)
        upperLayout.addWidget(addLaneButton)

        lowerLayout = QVBoxLayout()
        lowerLayout.addWidget(delLaneButton)
        lowerLayout.addWidget(saveButton)

        # set the layout
        layout = QHBoxLayout()
        layout.addWidget(self.canvas)
        layout.addLayout(upperLayout)
        layout.addLayout(lowerLayout)
        self.setLayout(layout)
        """
        
        # 점 저장용 리스트
        self.lane_points = []
        # 곡선별 점 인덱스 범위 저장 ([(start_idx, end_idx), ...])
        self.lane_labels = []
        # 점/곡선 아티스트 저장
        self.lane_point_artists = []  # scatter 반환값들 (Add Label 전 점)
        self.lane_curve_artists = []  # plot 반환값들
        self.all_point_artists = []   # 전체 점 아티스트 (곡선 생성 후에도 유지)

        # mouse event filter 연결 (FigureCanvas 위젯에)
        # self.mouse_filter = QtMouseEventFilter()
        # self.canvas.installEventFilter(self.mouse_filter)
        # self.mouse_filter.clicked.connect(self.on_canvas_click)

        # __init__ 안에 추가
        self.canvas.mpl_connect('button_press_event', self.on_mpl_click)

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
        

        if "colors" in pcd.point:
            np_colors = pcd.point.colors.numpy()
            np_colors = (np_colors * 255).astype(np.uint8)
        else:
            # 기본 회색 (R, G, B)
            np_colors = np.full((np_points.shape[0], 3), 128, dtype=np.uint8)

        vtk_points = vtk.vtkPoints()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
        
        # Normalize intensity to [0, 1]
        # np_colors = pcd.point.colors.numpy() 
        
        for pt, rgb in zip(np_points, np_colors):
            vtk_points.InsertNextPoint(*pt)
            vtk_colors.InsertNextTuple3(*rgb)

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


    def addPolyLine(self, line_points, color=(1,0,0), width=2):
        """
        line_points: [(x,y,z), …] 형태의 3D 좌표 리스트
        color: RGB 튜플 (0~1)
        width: 선 굵기 (픽셀)
        """
        # 1) vtkPoints 에 3D 점 삽입
        vtk_pts = vtk.vtkPoints()
        for i, (x,y,z) in enumerate(line_points):
            vtk_pts.InsertNextPoint(x, y, z)

        # 2) vtkPolyLine 으로 셀 구성
        polyLine = vtk.vtkPolyLine()
        polyLine.GetPointIds().SetNumberOfIds(len(line_points))
        for i in range(len(line_points)):
            polyLine.GetPointIds().SetId(i, i)

        cells = vtk.vtkCellArray()
        cells.InsertNextCell(polyLine)

        # 3) vtkPolyData 에 점과 라인 정보 설정
        lineData = vtk.vtkPolyData()
        lineData.SetPoints(vtk_pts)
        lineData.SetLines(cells)

        # 4) Mapper & Actor 생성
        lineMapper = vtk.vtkPolyDataMapper()
        lineMapper.SetInputData(lineData)

        lineActor = vtk.vtkActor()
        lineActor.SetMapper(lineMapper)
        lineActor.GetProperty().SetColor(*color)
        lineActor.GetProperty().SetLineWidth(width)

        # 5) 렌더러에 추가
        self.vtkRenderer.AddActor(lineActor)
        # 카메라 리셋은 선택사항
        # self.vtkRenderer.ResetCamera()

    def updatePointCloud(self, pcd_root='000000.pcd', index=0):
        pcd_file = self.list_img_path[index].replace(".jpg", ".pcd").replace(".png", ".pcd")
        full_path = os.path.join(pcd_root, pcd_file)

        if not os.path.exists(full_path):
            print("PCD not found:", full_path)
            return

        pcd = o3d.io.read_point_cloud(full_path)
        np_points = np.asarray(pcd.points)

        vtk_points = vtk.vtkPoints()
        for pt in np_points:
            vtk_points.InsertNextPoint(pt[0], pt[1], pt[2])

        polydata = vtk.vtkPolyData()
        polydata.SetPoints(vtk_points)

        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()

        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(vertex_filter.GetOutput())

        self.vtk_actor.SetMapper(mapper)
        self.vtk_renderer.ResetCamera()
        self.vtkWidget.GetRenderWindow().Render()

        
    def fastDraw(self, canvas):
        ''' for faster scene update '''
        self.canvas.setUpdatesEnabled(False)
        canvas.draw()
        self.canvas.setUpdatesEnabled(True)

    def loadFirstLabel(self):
        self.isLabelExist(self.img_path, self.imgIndex)

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
        # elif self.polygonRadio1.isChecked():
        #     self.beforeRadioChecked = self.polygonRadio1
        # elif self.polygonRadio2.isChecked():
        #     self.beforeRadioChecked = self.polygonRadio2
        # elif self.polygonRadio3.isChecked():
        #     self.beforeRadioChecked = self.polygonRadio3
        # elif self.objectRadio1.isChecked():
        #     self.beforeRadioChecked = self.objectRadio1
        # elif self.objectRadio2.isChecked():
        #     self.beforeRadioChecked = self.objectRadio2
        # elif self.objectRadio3.isChecked():
        #     self.beforeRadioChecked = self.objectRadio3
        # elif self.pointRadio1.isChecked():
        #     self.beforeRadioChecked = self.pointRadio1
        else:
            print("outofrange at radiobutton")

    def plotBackGround(self,img_path,action,isFirst=False):
        ''' Plot background method '''
        isPlot = True
        isEdge = False

        """
        if not isFirst:
            # if not saved, popup message box
            if self.imgIndex == len(self.list_img_path):
                self.saveFlag = self.isPosNotChange(img_path,self.imgIndex-1)
            else:
                self.saveFlag = self.isPosNotChange(img_path,self.imgIndex)

            if not self.saveFlag:
                isPlot = self.msgBoxEvent()
        """

        if isPlot:
            # increase img index
            if action == 0 and self.imgIndex < len(self.list_img_path):
                if self.imgIndex == -1 and isFirst == False :    # boundary scenario
                    self.imgIndex += 1
                self.imgIndex += 1
            elif action == 1 and self.imgIndex > -1:
                if self.imgIndex == len(self.list_img_path):
                    self.imgIndex -= 1
                self.imgIndex -= 1

            if self.imgIndex == len(self.list_img_path) or (isFirst == False and self.imgIndex == -1):
                isEdge = self.msgBoxReachEdgeEvent()

            if not isEdge:
                # for faster scene update
                self.canvas.setUpdatesEnabled(False)

                # clean up list points
                self.delAllLine()


                #while self.list_points:
                #    self.delLastLine()

                path = img_path + '/' + self.list_img_path[self.imgIndex].replace('\\', '/')

                # TODO: written for test
                img = cv2.imread(path)
                if img is None:
                    print(f"이미지 {path}를 불러올 수 없습니다.")
                    return

                # distortion 보정
                img_undistorted = cv2.undistort(img, self.k, self.distortion)

                # matplotlib 등에서 RGB로 쓰려면 변환
                img_undistorted = cv2.cvtColor(img_undistorted, cv2.COLOR_BGR2RGB)

                height, width, channels = img_undistorted.shape
                #width = int(width / 2)
                #height = int(height / 2)
                #self.resize(width,height)
                #self.showFullScreen()
                #self.showMaximized()

                if isFirst:
                    self.pyt = self.axes.imshow(img_undistorted)
                else:
                    self.pyt.set_data(img_undistorted)

                # initial (x,y) position in image range
                global x1,y1,x2,y2,x3,y3,x4,y4
                x_range = width - 0
                y_range = height - 0

                self.canvasSize = [width, height]

                # Divide into six equal segments
                x1,x2,x3,x4 = x_range*(4.0/6), x_range*(3.0/6), x_range*(2.0/6), x_range*(1.0/6)
                y1,y2,y3,y4 = y_range*(1.0/6), y_range*(2.0/6), y_range*(3.0/6), y_range*(4.0/6)

                # Produce an height*width black basemap
                #self.basemap = np.zeros([height,width,1], dtype=np.uint8)

                # If label text exist, draw previous output
                # TODO: read from existing label=
                if not isFirst:
                    isLabel = self.isLabelExist(img_path,self.imgIndex)

                #isLabel = True
                #if not isLabel and not isFirst:
                #    self.isLabelExist(img_path,self.imgIndex-1)

                # Edit window title
                self.setWindowTitle("Road Labeling Tool: " + self.list_img_path[self.imgIndex])

                self.canvas.draw()

                # for faster scene update
                self.canvas.setUpdatesEnabled(True)
                self.vtkRenderer.RemoveAllViewProps() 
                self.addPointCloudToVTK(self.imgIndex)

    def loadNextImage(self):
        self.saveAll(self.img_path)
        self.plotBackGround(self.img_path,0)

    def loadPrevImage(self):
        self.saveAll(self.img_path)
        self.plotBackGround(self.img_path,1)


    def plotDraggableLine(self, lineType, lineColor, verts=[], codes=[]):
        ''' Plot and define the 2 draggable points of the baseline '''
        if len(verts) == 0 or len(codes) == 0:
            verts = [(x1/2+x2/2, y1/2+y2/2), (x3/2+x4/2, y3/2+y4/2),]
            #codes = [Path.MOVETO,Path.CURVE4,Path.CURVE4,Path.CURVE4,]
            codes = [Path.MOVETO, Path.LINETO, ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', alpha=1, edgecolor=lineColor, lw=4, linestyle=lineType)

        self.axes.add_patch(patch)

        dr = DraggablePoint(patch, "line")
        self.list_points.append(dr)

    def plotDraggablePolygon(self, faceColor, verts=[], codes=[]):
        ''' Plot and define the Polygon '''
        if len(verts) == 0 or len(codes) == 0:
            verts = [(x3, y3), (x3, y4), (x4, y4), (x4, y3), (x3, y3), ]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor=faceColor, alpha=0.5, linestyle='-', lw=0)

        self.axes.add_patch(patch)

        dr = DraggablePoint(patch, "polygon")
        self.list_points.append(dr)

    def plotDraggableObject(self, lineColor, verts=[], codes=[]):
        ''' Plot and define the Polygon '''
        if len(verts) == 0 or len(codes) == 0:
            verts = [(x1, y1), (x1, y2), (x2, y2), (x2, y1), (x1, y1), ]
            codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]
        path = Path(verts, codes)
        patch = patches.PathPatch(path, facecolor='none', alpha=0.75, edgecolor=lineColor, lw=3, linestyle='-')

        self.axes.add_patch(patch)

        dr = DraggablePoint(patch, "object")
        self.list_points.append(dr)

    def loadImg(self, directory):
        try:
            self.list_img_path = []
            for file in os.listdir(os.path.join(directory, 'image')):
                if file.endswith('.jpg') or file.endswith('.png'):
                    print(file)
                    self.list_img_path.append(os.path.join('image', file))
            print("\ntotal", len(self.list_img_path), "files loaded\n")
        except Exception as e:
            sys.exit(str(e))

    def showPosition(self):
        ''' display current 4 points position '''
        text = ""
        for lineType, pts in zip(self.list_points_type,self.list_points):
            print(lineType,pts.get_position())
            text = text + lineType
            for index, (x, y) in enumerate(pts.get_position()):
                text = text + " " + str(x) + " " + str(y)
            text = text + "\n"
        print("")

        self.editBox.setPlainText(text)

    def addNewLine(self,select):
        ''' add a new line to figure '''
        lineType = ''
        lineColor = ''

        if select == 1:
            lineType = '-'
            lineColor = 'c'
            self.list_points_type.append('White')
        elif select == 2:
            lineType = '--'
            lineColor = 'c'
            self.list_points_type.append('WhiteDash')
        elif select == 3:
            lineType = '-'
            lineColor = 'm'
            self.list_points_type.append('Yellow')

        if select != 0:
            self.plotDraggableLine(lineType,lineColor)
            self.butconnect()

    def addNewLine(self):
        ''' add a new line to figure '''

        select = 0
        if self.beforeRadioChecked == self.lineRadio1:
            lineType = '-'
            lineColor = 'orangered'
            self.list_points_type.append('Yellow')
            self.plotDraggableLine(lineType, lineColor)
            select = 1
        elif self.beforeRadioChecked == self.lineRadio2:
            lineType = '-'
            lineColor = 'deepskyblue'
            self.list_points_type.append('White')
            self.plotDraggableLine(lineType, lineColor)
            select = 2
        elif self.beforeRadioChecked == self.lineRadio3:
            lineType = '--'
            lineColor = 'deepskyblue'
            self.list_points_type.append('WhiteDash')
            self.plotDraggableLine(lineType, lineColor)
            select = 3
        elif self.beforeRadioChecked == self.polygonRadio1:
            faceColor = 'mediumblue'
            self.list_points_type.append('Crosswalk')
            self.plotDraggablePolygon(faceColor)
            select = 4
        elif self.beforeRadioChecked == self.polygonRadio2:
            faceColor = 'blueviolet'
            self.list_points_type.append('StopLine')
            self.plotDraggablePolygon(faceColor)
            select = 5
        elif self.beforeRadioChecked == self.polygonRadio3:
            faceColor = 'saddlebrown'
            self.list_points_type.append('SpeedDump')
            self.plotDraggablePolygon(faceColor)
            select = 6
        elif self.beforeRadioChecked == self.objectRadio1:
            lineColor = 'palevioletred'
            self.list_points_type.append('Arrow')
            self.plotDraggableObject(lineColor, verts, codes)
            select = 7
        elif self.beforeRadioChecked == self.objectRadio2:
            lineColor = 'yellow'
            self.list_points_type.append('Diamond')
            self.plotDraggableObject(lineColor, verts, codes)
            select = 8
        elif self.beforeRadioChecked == self.objectRadio3:
            lineColor = 'limegreen'
            self.list_points_type.append('RoadSign')
            self.plotDraggableObject(lineColor, verts, codes)
            select = 9
        elif self.beforeRadioChecked == self.pointRadio1:
            select = 10
            pass
        else:
            print("outofrange at addNewLine")

        if select != 0:
            # for faster scene update
            self.canvas.setUpdatesEnabled(False)
            self.butconnect()
            # for faster scene update
            self.canvas.setUpdatesEnabled(True)


    def delLastLine(self):
        ''' del the last line to figure '''
        if self.list_points:
            # for faster scene update
            self.canvas.setUpdatesEnabled(False)
            self.butdisconnect()
            self.list_points[-1].line.remove()
            self.list_points.pop()
            self.butconnect()
            # for faster scene update
            self.canvas.setUpdatesEnabled(True)

        if self.axes.patches:
            self.axes.patches[-1].remove()

        if self.list_points_type:
            self.list_points_type.pop()

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

    def addNewPoint(self):
        ''' add a new line points to figure '''
        if self.list_points_type[-1] in ['Yellow', 'White', 'WhiteDash']:
            lineType = self.list_points_type[-1]
            pts = self.list_points[-1]
            pos = pts.get_position()
            verts = [(256 + random.randrange(0,512), 256 + random.randrange(0,512))]
            for index, (x, y) in enumerate(pos):
                verts.append((x, y))
            codes = [Path.MOVETO, ]
            for i in range(len(pos)):
                codes.append(Path.LINETO)

            lineType = ''
            lineColor = ''
            
            if self.list_points_type[-1] == 'Yellow':
                lineType = '-'
                lineColor = 'orangered'
                self.list_points_type.append('Yellow')
            elif self.list_points_type[-1] == 'White':
                lineType = '-'
                lineColor = 'deepskyblue'
                self.list_points_type.append('White')
            elif self.list_points_type[-1] == 'WhiteDash':
                lineType = '--'
                lineColor = 'deepskyblue'
                self.list_points_type.append('WhiteDash')

            self.delLastLine()
            self.plotDraggableLine(lineType, lineColor, verts, codes)
            # for faster scene update
            self.canvas.setUpdatesEnabled(False)
            self.butconnect()
            # for faster scene update
            self.canvas.setUpdatesEnabled(True)

        elif self.list_points_type[-1] in ['Crosswalk', 'StopLine', 'SpeedDump']:
            lineType = self.list_points_type[-1]
            pts = self.list_points[-1]
            pos = pts.get_position()
            verts = []
            for index, (x, y) in enumerate(pos):
                verts.append((x, y))
            verts.insert(1, (256 + random.randrange(0,512), 256 + random.randrange(0,512)))
            codes = [Path.MOVETO, ]
            for i in range(len(pos)-1):
                codes.append(Path.LINETO)
            codes.append(Path.CLOSEPOLY)

            faceColor = ''

            if self.list_points_type[-1] == 'Crosswalk':
                faceColor = 'mediumblue'
                self.list_points_type.append('Crosswalk')
            elif self.list_points_type[-1] == 'StopLine':
                faceColor = 'blueviolet'
                self.list_points_type.append('StopLine')
            elif self.list_points_type[-1] == 'SpeedDump':
                faceColor = 'saddlebrown'
                self.list_points_type.append('SpeedDump')

            self.delLastLine()
            self.plotDraggablePolygon(faceColor, verts, codes)
            # for faster scene update
            self.canvas.setUpdatesEnabled(False)
            self.butconnect()
            # for faster scene update
            self.canvas.setUpdatesEnabled(True)

        else:
            print("Point Type Not Supported")


    def delLastPoint(self):
        ''' del the last line points to figure '''
        # TODO: CODING
        if self.list_points_type[-1] in ['Yellow', 'White', 'WhiteDash']:
            lineType = self.list_points_type[-1]
            pts = self.list_points[-1]
            pos = pts.get_position()

            if len(pos) <= 2:
                self.delLastLine()
                return

            verts = []
            for index, (x, y) in enumerate(pos):
                verts.append((x, y))
            verts.pop(0)
            codes = [Path.MOVETO, ]
            for i in range(len(pos)-2):
                codes.append(Path.LINETO)

            lineType = ''
            lineColor = ''

            if self.list_points_type[-1] == 'Yellow':
                lineType = '-'
                lineColor = 'orangered'
                self.list_points_type.append('Yellow')
            elif self.list_points_type[-1] == 'White':
                lineType = '-'
                lineColor = 'deepskyblue'
                self.list_points_type.append('White')
            elif self.list_points_type[-1] == 'WhiteDash':
                lineType = '--'
                lineColor = 'deepskyblue'
                self.list_points_type.append('WhiteDash')

            self.delLastLine()
            self.plotDraggableLine(lineType, lineColor, verts, codes)
            # for faster scene update
            self.canvas.setUpdatesEnabled(False)
            self.butconnect()
            # for faster scene update
            self.canvas.setUpdatesEnabled(True)

        elif self.list_points_type[-1] in ['Crosswalk', 'StopLine', 'SpeedDump']:
            lineType = self.list_points_type[-1]
            pts = self.list_points[-1]
            pos = pts.get_position()

            if len(pos) <= 3:
                self.delLastLine()
                return

            verts = []
            for index, (x, y) in enumerate(pos):
                verts.append((x, y))
            verts.pop(1)
            codes = [Path.MOVETO, ]
            for i in range(len(pos) - 3):
                codes.append(Path.LINETO)
            codes.append(Path.CLOSEPOLY)

            faceColor = ''

            if self.list_points_type[-1] == 'Crosswalk':
                faceColor = 'mediumblue'
                self.list_points_type.append('Crosswalk')
            elif self.list_points_type[-1] == 'StopLine':
                faceColor = 'blueviolet'
                self.list_points_type.append('StopLine')
            elif self.list_points_type[-1] == 'SpeedDump':
                faceColor = 'saddlebrown'
                self.list_points_type.append('SpeedDump')

            self.delLastLine()
            self.plotDraggablePolygon(faceColor, verts, codes)
            # for faster scene update
            self.canvas.setUpdatesEnabled(False)
            self.butconnect()
            # for faster scene update
            self.canvas.setUpdatesEnabled(True)

        else:
            print("Point Type Not Supported")

    def butconnect(self):
        ''' connect current DraggablePoints '''
        for pts in self.list_points:
            pts.connect()
        self.canvas.draw()
        #self.fastDraw(self.canvas)

    def butdisconnect(self):
        ''' disconnect current DraggablePoints '''
        for pts in self.list_points:
            pts.disconnect()
        self.canvas.draw()
        #self.fastDraw(self.canvas)

    def savePng(self,inputName,outputName):
        ''' save current figure to png '''
#        self.figure.savefig('test.png',bbox_inches='tight', pad_inches = 0)
        # Produce an height*width black basemap
        """
        self.basemap = np.zeros([self.canvasSize[1],self.canvasSize[0],1], dtype=np.uint8)
        with open(inputName+".txt", "r") as text_file:
            text_line = [text_line.rstrip('\n') for text_line in text_file]

        curvepoints = 20
        thickness = 3
        for item in text_line:
            pos = item.split(',')
            line_type, pos = pos[0],pos[1:]
            nodes = self.bezierCurve(pos,curvepoints)
            nodes = nodes.reshape((-1, 1, 2))

            if line_type == 'White':
                cv2.polylines(self.basemap, [nodes], False, (255, 0, 0), thickness)
            elif line_type == 'WhiteDash':
                cv2.polylines(self.basemap, [nodes], False, (200, 0, 0), thickness)
            elif line_type == 'Yellow':
                cv2.polylines(self.basemap, [nodes], False, (150, 0, 0), thickness)

#            cv2.imshow('image',self.basemap)

        cv2.imwrite(outputName, self.basemap)
#        x = cv2.imread(outputName)
#        print x[np.nonzero(x)]
        """

    def saveText(self,outputName):
        ''' save line type and positions to txt '''
        with open(outputName+".txt", "w") as text_file:
            for lineType, pts in zip(self.list_points_type,self.list_points):
                pos = pts.get_position()
                text_file.write("%s " % lineType)
                for index, (x,y) in enumerate(pos):
                    text_file.write("%s %s" % (x, y))
                    if index != len(pos)-1:
                        text_file.write(" ")
                text_file.write("\n")

        self.saveFlag = True

    def saveAll(self,img_path):
        ''' save text and save png '''
        ind = self.imgIndex
        if self.imgIndex == len(self.list_img_path):
            ind = ind - 1

        if self.imgIndex == -1:
            ind = ind + 1

        # f1 = img_path+"label_txt"+os.sep+self.list_img_path[ind][:-4]
        # f2 = img_path+"label_png"+os.sep+self.list_img_path[ind][:-4]+".png"
        # TODO: fix export png
        f1 = img_path + os.sep + self.list_img_path[ind][:-4]
        self.saveText(f1)

        #self.savePng(f1,f2)

    # def loadPrevLabel(self):
    #     ''' load previous labels '''
    #     ind = self.imgIndex
    #     if self.imgIndex == len(self.list_img_path):
    #         ind = ind - 1

    #     if self.imgIndex == -1:
    #         ind = ind + 1

    #     status = self.isLabelExist(self.img_path, ind-1)

    def msgBoxEvent(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Warning)
        msgBox.setWindowTitle('WARNING')
        msgBox.setText( "Your changes have not been saved.\nAre you sure you want to discard the changes?" )
        msgBox.setInformativeText( "Press OK to continue, or Cancel to stay on the current page." )
        msgBox.addButton( QMessageBox.Ok )
        msgBox.addButton( QMessageBox.Cancel )

        msgBox.setDefaultButton( QMessageBox.Cancel )
        ret = msgBox.exec_()

        if ret == QMessageBox.Ok:
            if self.imgIndex == len(self.list_img_path):
                self.saveFlag = False
            else:
                self.saveFlag = True
            return True
        else:
            return False

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

    def isPosNotChange(self,img_path,index):
        # load label txt
        """
        fileName = img_path+"label_txt"+os.sep+self.list_img_path[index][:-4]+".txt"
        lab = []
        labnp = np.array(lab)
        curnp = np.array(lab)
        try:
            with open(fileName, 'r') as f:
                x = f.read().splitlines()
                for line in x:
                    select,xstr1,ystr1,xstr2,ystr2,xstr3,ystr3,xstr4,ystr4 = line.split(',')
                    x1,y1,x2,y2,x3,y3,x4,y4 = float(xstr1),float(ystr1),float(xstr2),float(ystr2),float(xstr3),float(ystr3),float(xstr4),float(ystr4)
                    lab.append([x1,y1,x2,y2,x3,y3,x4,y4])
                labnp = np.array(lab)

        except IOError:
            lab = []

        # current dp
        if self.list_points:
            curnp = self.list_points[0].get_position()
            for pts in range(1,len(self.list_points)):
                curnp = np.vstack((curnp,self.list_points[pts].get_position()))
            curnp = curnp.reshape(-1,8)

        # check xdim
        if labnp.shape[0] != curnp.shape[0]:
            return False

        # check content
        for l in curnp:
            if l not in labnp:
                return False
        """
        return True

    def isLabelExist(self,img_path,index):
        fileName = img_path+os.sep+self.list_img_path[index][:-4]+".txt"
        select = ''
        try:
            with open(fileName, 'r') as f:
                x = f.read().splitlines()
                # global x1,y1,x2,y2,x3,y3,x4,y4
                # tmpx1, tmpy1, tmpx2, tmpy2, tmpx3, tmpy3, tmpx4, tmpy4 = x1,y1,x2,y2,x3,y3,x4,y4
                for line in x:
                    elemlist = line.split(' ')
                    category = elemlist[0]
                    points = elemlist[1:]

                    verts = [(points[i*2], points[i*2+1]) for i in range(int(len(points)/2))]
                    codes = []

                    if category in ['Yellow', 'White', 'WhiteDash']:
                        codes = [Path.MOVETO, ]
                        for i in range(len(verts)-1):
                            codes.append(Path.LINETO)
                    elif category in ['Crosswalk', 'StopLine', 'SpeedDump']:
                        codes = [Path.MOVETO, ]
                        for i in range(len(verts) - 2):
                            codes.append(Path.LINETO)
                        codes.append(Path.CLOSEPOLY)
                    elif category in ['Arrow', 'Diamond', 'RoadSign']:
                        codes = [Path.MOVETO, Path.LINETO, Path.LINETO, Path.LINETO, Path.CLOSEPOLY, ]

                    if category == 'Yellow':
                        lineType = '-'
                        lineColor = 'orangered'
                        self.list_points_type.append('Yellow')
                        self.plotDraggableLine(lineType, lineColor, verts, codes)
                    elif category == 'White':
                        lineType = '-'
                        lineColor = 'deepskyblue'
                        self.list_points_type.append('White')
                        self.plotDraggableLine(lineType, lineColor, verts, codes)
                    elif category == 'WhiteDash':
                        lineType = '--'
                        lineColor = 'deepskyblue'
                        self.list_points_type.append('WhiteDash')
                        self.plotDraggableLine(lineType, lineColor, verts, codes)
                    elif category == 'Crosswalk':
                        faceColor = 'mediumblue'
                        self.list_points_type.append('Crosswalk')
                        self.plotDraggablePolygon(faceColor, verts, codes)
                    elif category == 'StopLine':
                        faceColor = 'blueviolet'
                        self.list_points_type.append('StopLine')
                        self.plotDraggablePolygon(faceColor, verts, codes)
                    elif category == 'SpeedDump':
                        faceColor = 'saddlebrown'
                        self.list_points_type.append('SpeedDump')
                        self.plotDraggablePolygon(faceColor, verts, codes)
                    elif category == 'Arrow':
                        lineColor = 'palevioletred'
                        self.list_points_type.append('Arrow')
                        self.plotDraggableObject(lineColor, verts, codes)
                    elif category == 'Diamond':
                        lineColor = 'yellow'
                        self.list_points_type.append('Diamond')
                        self.plotDraggableObject(lineColor, verts, codes)
                    elif category == 'RoadSign':
                        lineColor = 'limegreen'
                        self.list_points_type.append('RoadSign')
                        self.plotDraggableObject(lineColor, verts, codes)
                    else:
                        pass

                    #self.butconnect()

                    # previous code\
                    """
                    select,xstr1,ystr1,xstr2,ystr2,xstr3,ystr3,xstr4,ystr4 = line.split(' ')
                    x1,y1,x2,y2,x3,y3,x4,y4 = float(xstr1),float(ystr1),float(xstr2),float(ystr2),float(xstr3),float(ystr3),float(xstr4),float(ystr4)

                    if select == 'White':
                        lineType = 1
                    elif select == 'WhiteDash':
                        lineType = 2
                    elif select == 'Yellow':
                        lineType = 3

                    self.addNewLine(lineType)
                    """

                self.butconnect()

                #x1,y1,x2,y2,x3,y3,x4,y4 = tmpx1, tmpy1, tmpx2, tmpy2, tmpx3, tmpy3, tmpx4, tmpy4
                self.saveFlag = True

            return True
        except IOError:
#            print "Could not read file:", fileName
            return False

    def bezierCurve(self,pos,num=2):
        ''' bezier Curve formula
        X = (1-t)^3A + 3t(1-t)^2B + 3t^2(1-t)C + t^3D
        '''
        x1,y1,x2,y2,x3,y3,x4,y4 = pos
        x1,y1,x2,y2,x3,y3,x4,y4 = float(x1),float(y1),float(x2),float(y2),float(x3),float(y3),float(x4),float(y4)
        T_list = np.arange(0., 1+1./num, 1./num)

        X = [ pow(1-t,3)*x1 + 3*t*pow(1-t,2)*x2 + 3*pow(t,2)*(1-t)*x3 + pow(t,3)*x4 for t in T_list]
        Y = [ pow(1-t,3)*y1 + 3*t*pow(1-t,2)*y2 + 3*pow(t,2)*(1-t)*y3 + pow(t,3)*y4 for t in T_list]
        return np.array(zip(X,Y),dtype=np.int32)

    def closeEvent(self, event):
        if self.imgIndex == len(self.list_img_path):
            self.saveFlag = self.isPosNotChange(self.img_path,self.imgIndex-1)
        else:
            self.saveFlag = self.isPosNotChange(self.img_path,self.imgIndex)

        if not self.saveFlag:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Warning)
            msgBox.setWindowTitle('WARNING')
            msgBox.setText( "Your changes have not been saved.\nAre you sure you want to discard the changes?" )
            msgBox.setInformativeText( "Press OK to exit, or Cancel to stay on the current page." )
            msgBox.setStandardButtons(QMessageBox.Cancel | QMessageBox.Ok);
            msgBox.setDefaultButton(QMessageBox.Cancel);
            if msgBox.exec_() == QMessageBox.Ok:
                event.accept()
            else:
                event.ignore()

    def onclick(self,event):
        print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f'%(event.button, event.x, event.y, event.xdata, event.ydata))

    def UI(self,path):
        self.label = QLabel(self)
        self.pixmap = QPixmap(os.getcwd()+"000000.jpg")
        self.label.setPixmap(self.pixmap)
        self.resize(self.pixmap.width(),self.pixmap.height())
        self.label.mousePressEvent = self.getPos
        self.show()

    def getPos(self,event):
        x = event.pos().x()
        y = event.pos().y()
        print('Click',(x,y))

    def on_canvas_click(self, x, y, button):
        if button != Qt.LeftButton:
            return
        # 픽셀 좌표(x, y) → figure 좌표 (0~1)
        width, height = self.canvas.get_width_height()
        x_fig = x / width
        y_fig = y / height

        # figure 좌표 → axes 좌표 (데이터 좌표)
        # matplotlib은 (0,0)이 왼쪽 아래, Qt는 (0,0)이 왼쪽 위이므로 y축 뒤집기 필요
        y_fig = 1 - y_fig

        # axes.transAxes는 (0,0)~(1,1) → 데이터 좌표 변환
        xdata, ydata = self.axes.transAxes.transform((x_fig, y_fig))
        xdata, ydata = self.axes.transData.inverted().transform((xdata, ydata))

        self.lane_points.append((xdata, ydata))
        self.axes.plot(xdata, ydata, 'o', color='blue')
        self.canvas.draw()

    def draw_lane_curve(self):
        if len(self.lane_points) < 2:
            return
        xs, ys = centripetal_catmull_rom(self.lane_points)
        if self.beforeRadioChecked == self.lineRadio1:  # Yellow Line
            color = 'yellow'
            linestyle = '-'
        elif self.beforeRadioChecked == self.lineRadio2:  # White Line
            color = '#ff69b4'  # 핫핑크
            linestyle = '-'
        elif self.beforeRadioChecked == self.lineRadio3:  # White Dash Line
            color = '#ff69b4'  # 핫핑크
            linestyle = '--'
        else:
            color = 'limegreen'
            linestyle = '-'
        curve_artist, = self.axes.plot(xs, ys, linestyle, color=color)
        self.lane_curve_artists.append(curve_artist)
        # 곡선 생성에 사용된 점들의 인덱스 기록
        n = len(self.lane_points)
        if self.lane_labels:
            last_end = self.lane_labels[-1][1]
        else:
            last_end = 0
        self.lane_labels.append((last_end, last_end + n))
        used_points = self.lane_points.copy()  # 복사!
        # 점 리스트/아티스트 초기화
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

        lane_points_3d = lane_points_3d_from_pcd_and_lane(
            rgb_image,
            np_points,
            self.k,
            self.r,
            self.t,
            lane_polyline_img_coords,
            lane_thickness=10
        )
        # points_in_camera_homo 계산
        if lane_points_3d.shape[0] == 4:
            points_3d = lane_points_3d[:3, :].T
        else:
            points_3d = lane_points_3d.T


        self.addLanePointsToVTK(points_3d, color=(1,0,0), size=3)

    def on_mpl_click(self, event):
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.lane_points.append((event.xdata, event.ydata))
        if self.beforeRadioChecked == self.lineRadio1:  # Yellow Line
            edgecolor = 'yellow'
        elif self.beforeRadioChecked == self.lineRadio2 or self.beforeRadioChecked == self.lineRadio3:  # White/White Dash
            edgecolor = '#ff69b4'
        else:
            edgecolor = 'blue'
        artist = self.axes.scatter(
            event.xdata, event.ydata,
            s=60,
            facecolors='white',
            edgecolors=edgecolor,
            linewidths=2
        )
        self.lane_point_artists.append(artist)
        self.all_point_artists.append(artist)
        self.canvas.draw()

    def delete_last_label(self):
        # 가장 최근 곡선과 해당 점들 삭제
        if not self.lane_labels or not self.lane_curve_artists:
            return
        # 곡선 삭제
        last_curve = self.lane_curve_artists.pop()
        last_curve.remove()
        # 점 삭제
        start, end = self.lane_labels.pop()
        try:
            # all_point_artists에서 해당 점 아티스트 remove 및 리스트에서 삭제
            for i in range(end-1, start-1, -1):
                artist = self.all_point_artists.pop(i)
                artist.remove()
        except Exception as e:
            print(f"[Delete Last Label] Error: {e}")
        self.canvas.draw()

    def delete_last_point(self):
        # Add Label(곡선) 생성 전, 가장 최근 점 하나만 삭제
        if not self.lane_point_artists or not self.lane_points:
            return
        last_artist = self.lane_point_artists.pop()
        last_artist.remove()
        self.lane_points.pop()
        self.canvas.draw()

    def addLanePointsToVTK(self, points, color=(1,0,0), size=8):
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

        # 색상 지정
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


def genWindow(path):
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # 또는 'Windows', 'WindowsVista', 'Mac', ...
    
    qss = """
    QWidget {
        background-color: #232629;
        color: #f0f0f0;
        font-weight: bold;
    }
    QPushButton {
        background-color: #ff69b4;
        color: black;
        border-radius: 5px;
        padding: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #ff69b4;
    }
    QPushButton:pressed {
        background-color: #c2185b;
        color: white;
    }
    """
    app.setStyleSheet(qss)
    
    window = Window(path)
    window.showMaximized()
    window.show()
    sys.exit(app.exec_())

