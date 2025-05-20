# main.py
import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QPushButton, QGraphicsView, QGraphicsScene,
    QGraphicsPixmapItem, QGraphicsEllipseItem, QGraphicsPathItem
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen, QColor, QPainterPath, QPixmap
from mouse_event import QtMouseEventFilter
from polynomial import fit_parametric_poly, centripetal_catmull_rom

class ImageLabelerWindow(QMainWindow):
    def __init__(self, image_path: str):
        super().__init__()
        self.setWindowTitle("Lane Labeler")
        self.resize(800, 600)

        # 중앙 위젯 및 레이아웃
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # QGraphicsView/Scene 준비
        self.view = QGraphicsView()
        layout.addWidget(self.view)

        self.scene = QGraphicsScene(self)
        self.view.setScene(self.scene)

        # 배경 이미지
        pixmap = QPixmap(image_path)
        self.scene.addItem(QGraphicsPixmapItem(pixmap))

        # Add Curve 버튼
        self.btn_add = QPushButton("Add Curve")
        layout.addWidget(self.btn_add)

        # 라벨 점 저장
        self.points = []

        # 이벤트 필터 설치
        self.mouse_filter = QtMouseEventFilter()
        self.view.viewport().installEventFilter(self.mouse_filter)

        # 시그널 연결
        self.mouse_filter.clicked.connect(self.on_click)
        self.btn_add.clicked.connect(self.on_add_curve)

    def on_click(self, x: float, y: float, button: int):
        # 좌클릭만 처리
        if button != Qt.LeftButton:
            return
        # 씬 좌표로 변환 필요 없이 eventFilter gives viewport coords;
        # map to scene
        scene_pt = self.view.mapToScene(int(x), int(y))
        sx, sy = scene_pt.x(), scene_pt.y()

        # 파란 점 표시
        r = 5
        ellipse = QGraphicsEllipseItem(sx-r, sy-r, r*2, r*2)
        pen = QPen(QColor('blue'))
        pen.setWidth(2)
        ellipse.setPen(pen)
        from PyQt5.QtGui import QBrush
        ellipse.setBrush(QBrush())  # transparent brush
        self.scene.addItem(ellipse)

        self.points.append((sx, sy))
        print(f"Point added: ({sx:.1f}, {sy:.1f})")

    def on_add_curve(self):
        if len(self.points) < 2:
            print("Need at least 2 points to fit a curve.")
            return

        # 다항식 보간
        x_lin, y_lin = centripetal_catmull_rom(self.points)

        # QPainterPath로 곡선 그리기
        path = QPainterPath()
        path.moveTo(x_lin[0], y_lin[0])
        for x, y in zip(x_lin[1:], y_lin[1:]):
            path.lineTo(x, y)

        path_item = QGraphicsPathItem(path)
        pen = QPen(QColor('red'))
        pen.setWidth(2)
        path_item.setPen(pen)
        self.scene.addItem(path_item)

        print(f"Curve drawn through {len(self.points)} points.")
        self.points.clear()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ImageLabelerWindow('r.jpg')
    window.show()
    sys.exit(app.exec_())
