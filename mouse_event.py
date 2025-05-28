# qt_mouse_event.py
from PyQt5.QtCore import QObject, pyqtSignal, QEvent, Qt
from PyQt5.QtGui import QMouseEvent
from PyQt5.QtWidgets import QGraphicsView


class QtMouseEventFilter(QObject):
    
    clicked = pyqtSignal(float, float, int)
    doubleClicked = pyqtSignal(float, float, int)
    rightClicked = pyqtSignal(float, float)

    def eventFilter(self, obj, event):
        if isinstance(event, QMouseEvent):
            if event.type() == QEvent.MouseButtonRelease:
                x, y = event.x(), event.y()
                if event.button() == Qt.LeftButton:
                    self.clicked.emit(x, y, event.button())
                elif event.button() == Qt.RightButton:
                    self.rightClicked.emit(x, y)
            elif event.type() == QEvent.MouseButtonDblClick:
                x, y = event.x(), event.y()
                self.doubleClicked.emit(x, y, event.button())
        return False


class ClickableGraphicsView(QGraphicsView):
    clicked = pyqtSignal(float, float, int)
    doubleClicked = pyqtSignal(float, float)
    rightClicked = pyqtSignal(float, float)

    def mousePressEvent(self, event):
        if isinstance(event, QMouseEvent):
            scene_pt = self.mapToScene(event.pos())
            x, y = scene_pt.x(), scene_pt.y()
            if event.button() == Qt.LeftButton:
                self.clicked.emit(x, y, event.button())
            elif event.button() == Qt.RightButton:
                self.rightClicked.emit(x, y)
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event):
        if isinstance(event, QMouseEvent) and event.button() == Qt.LeftButton:
            scene_pt = self.mapToScene(event.pos())
            self.doubleClicked.emit(scene_pt.x(), scene_pt.y())
        super().mouseDoubleClickEvent(event)

    def on_mpl_click(self, event):
        if event.button != 1:  # 1: left click
            return
        if event.xdata is None or event.ydata is None:
            return
        self.lane_points.append((event.xdata, event.ydata))
        self.axes.plot(event.xdata, event.ydata, 'o', color='blue')
        self.canvas.draw()
