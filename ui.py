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
import json
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
from converter import load_calibration_params, lane_points_3d_from_pcd_and_lane, lidar_points_in_image, sample_lane_points

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
    # Lane 색상 및 타입 매핑
    LANE_COLORS = {
        'Yellow': {
            'mpl_color': 'red',
            'vtk_color': (1.0, 0.0, 0.0),
            'category': 'yellow_lane'
        },
        'White': {
            'mpl_color': '#ff69b4',
            'vtk_color': (1.0, 0.412, 0.706),
            'category': 'white_lane'
        },
        'WhiteDash': {
            'mpl_color': 'limegreen',
            'vtk_color': (0.196, 0.804, 0.196),
            'category': 'white_dash_lane'
        },
        'Default': {
            'mpl_color': 'limegreen',
            'vtk_color': (0.196, 0.804, 0.196),
            'category': 'unknown'
        }
    }

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
        self.sampled_pts = []
        self.sampled_uv = []

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

        lineGroupBox = QGroupBox("Line", self)


        # 라디오 버튼 스타일 시트
        radio_style = """
            QRadioButton {
                spacing: 10px;
                font-weight: bold;
                min-height: 30px;
            }
            QRadioButton::indicator {
                width: 15px;
                height: 15px;
                border-radius: 10px;
            }
            QRadioButton::indicator::unchecked {
                background-color: #cccccc;
                border: 2px solid #666666;
            }
            QRadioButton::indicator::checked {
                background-color: qlineargradient(spread:pad, x1:0, y1:0, x2:0.857143, y2:0.857955, stop:0 rgba(10, 242, 251, 255), stop:1 rgba(224, 6, 159, 255));
                border: 2px solid qlineargradient(spread:pad, x1:0, y1:0, x2:0.857143, y2:0.857955, stop:0 rgba(10, 242, 251, 255), stop:1 rgba(224, 6, 159, 255));
            }
            QRadioButton::indicator::unchecked:hover {
                background-color: #ff69b4;
            }
            QRadioButton::indicator::checked:hover {
                background-color: #ff69b4;
            }
        """

        # Yellow Line 라디오 버튼
        self.lineRadio1 = QRadioButton("Yellow Line", self)
        self.lineRadio1.setStyleSheet(radio_style)
        self.lineRadio1.setChecked(True)
        self.beforeRadioChecked = self.lineRadio1
        self.lineRadio1.clicked.connect(self.radioButtonClicked)

        # White Line 라디오 버튼
        self.lineRadio2 = QRadioButton("White Line", self)
        self.lineRadio2.setStyleSheet(radio_style)
        self.lineRadio2.clicked.connect(self.radioButtonClicked)

        # White Dash Line 라디오 버튼
        self.lineRadio3 = QRadioButton("White Dash Line", self)
        self.lineRadio3.setStyleSheet(radio_style)
        self.lineRadio3.clicked.connect(self.radioButtonClicked)

        addLaneButton = QPushButton("Add Label")
        addLaneButton.clicked.connect(self.draw_lane_curve)

        delLaneButton = QPushButton("Delete Label")
        delLaneButton.clicked.connect(self.delete_last_label)

        delPointButton = QPushButton("Delete Point")
        delPointButton.clicked.connect(self.delete_last_point)

        curPosButton = QPushButton("Show current Labels")
        curPosButton.clicked.connect(self.showPosition)

        verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        nextImgButton = QPushButton("Next Image")

        nextImgButton.clicked.connect(self.loadNextImage)

        preImgButton = QPushButton("Prev Image")

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

        addDelLayout = QHBoxLayout()
        addDelLayout.addWidget(delPointButton)

        addLayout = QVBoxLayout()
        addLayout.addWidget(addLaneLabel)
        addLayout.addWidget(lineGroupBox)

        addLayout.addWidget(addLaneButton)
        addLayout.addWidget(delLaneButton)
        addLayout.addLayout(addDelLayout)

        rightLayout = QVBoxLayout()
        rightLayout.addLayout(addLayout)
        rightLayout.addSpacing(20)
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

        # 점 저장용 리스트
        self.lane_points = []
        # 곡선별 점 인덱스 범위 저장 ([(start_idx, end_idx), ...])
        self.lane_labels = []
        # 점/곡선 아티스트 저장
        self.lane_point_artists = []  # scatter 반환값들 (Add Label 전 점)
        self.lane_curve_artists = []  # plot 반환값들
        self.all_point_artists = []   # 전체 점 아티스트 (곡선 생성 후에도 유지)

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


        if hasattr(self, 'lane_curve_artists'):
            for curve in self.lane_curve_artists:
                curve.remove()
            self.lane_curve_artists.clear()

        # 점 아티스트도 함께 제거 (선택 사항)
        if hasattr(self, 'all_point_artists'):
            for pt in self.all_point_artists:
                pt.remove()
            self.all_point_artists.clear()
                
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

                # Edit window title
                self.setWindowTitle("IRCV: 3D Lane Labeling Tool")

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
        ''' display current labeled lanes '''
        text = "Current Labeled Lanes:\n\n"
        
        # 라인 정보 표시
        if hasattr(self, 'sampled_uv') and self.sampled_uv is not None:
            for i, lane_type in enumerate(self.list_points_type):
                text += f"{i+1} Lane: {lane_type}\n"
                text += f"\nTotal number of labeled lanes: {len(self.list_points_type)}\n"

        self.editBox.setPlainText(text)




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
                lineColor = '#ff69b4'
                self.list_points_type.append('White')
            elif self.list_points_type[-1] == 'WhiteDash':
                lineType = '--'
                lineColor = 'red'
                self.list_points_type.append('WhiteDash')

            self.delLastLine()
            self.plotDraggableLine(lineType, lineColor, verts, codes)
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

    def saveAll(self, img_path):
        ''' Save lane data in the specified format '''
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
        if hasattr(self, 'sampled_pts') and self.sampled_pts is not None:
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

        lane_points_3d, points_2d = lane_points_3d_from_pcd_and_lane(
            rgb_image,
            np_points,
            self.k,
            self.r,
            self.t,
            lane_polyline_img_coords,
            lane_thickness=3
        )
        # points_in_camera_homo 계산
        if lane_points_3d.shape[0] == 4:
            points_3d = lane_points_3d[:3, :].T
        else:
            points_3d = lane_points_3d.T

        # lane points 누적
        new_sampled_pts, new_sampled_uv = sample_lane_points(points_3d.T, points_2d, num_samples=20)
        print(f"새로운 sampled point 개수: {len(new_sampled_pts)}")
        
        # 각 lane별로 points와 uv를 저장
        self.sampled_pts.append(new_sampled_pts)
        self.sampled_uv.append(new_sampled_uv)
        
        # 모든 lane의 points와 uv를 합쳐서 VTK에 추가
        all_points = np.vstack(self.sampled_pts)
        all_uv = np.vstack(self.sampled_uv)
        print(f"누적된 sampled point 총 개수: {len(all_points)}")
        self.addLanePointsToVTK(new_sampled_pts, color=vtk_color, size=5)
        
    def on_mpl_click(self, event):
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        self.lane_points.append((event.xdata, event.ydata))
        lane_type = None
        if self.beforeRadioChecked == self.lineRadio1:  # Yellow Line
            lane_type = 'Yellow'
        elif self.beforeRadioChecked == self.lineRadio2:  # White Line
            lane_type = 'White'
        elif self.beforeRadioChecked == self.lineRadio3:  # White Dash Line
            lane_type = 'WhiteDash'
        
        lane_info = self.LANE_COLORS.get(lane_type, self.LANE_COLORS['Default'])
        edgecolor = lane_info['mpl_color']
        
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

        # VTK에서 마지막 레인 Actor 삭제
        if hasattr(self, 'lane_vtk_actors') and self.lane_vtk_actors:
            last_actor = self.lane_vtk_actors.pop()
            self.vtkRenderer.RemoveActor(last_actor)
            self.vtkWidget.GetRenderWindow().Render()

        self.canvas.draw()

    def delete_last_point(self):
        # Add Label(곡선) 생성 전, 가장 최근 점 하나만 삭제
        if not self.lane_point_artists or not self.lane_points:
            return
        last_artist = self.lane_point_artists.pop()
        last_artist.remove()
        self.lane_points.pop()
        self.canvas.draw()

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

        # 생성된 레인 Actor 저장
        if not hasattr(self, 'lane_vtk_actors'):
            self.lane_vtk_actors = []
        self.lane_vtk_actors.append(actor)

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
        color: white;
        border-radius: 5px;
        padding: 5px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #2080E1;
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

