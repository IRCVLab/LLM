import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QMouseEvent

class EventTools:
    """
    Mixin for event handling methods (mouse, keyboard, matplotlib, etc)
    Assumes main Window class initializes all shared state and UI components.
    """
    def on_mpl_click(self, event):
        # Matplotlib canvas click event
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

    # If you have other event handlers (mouse/key events, etc.), add them here as methods.
    # Example stubs:
    # def mousePressEvent(self, event):
    #     ...
    # def keyPressEvent(self, event):
    #     ...


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

    def delLastLine(self):
        ''' del the last line to figure '''
        if self.list_points:
            self.canvas.setUpdatesEnabled(False)
            self.butdisconnect()
            self.list_points[-1].line.remove()
            self.list_points.pop()
            self.butconnect()
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

            self.canvas.setUpdatesEnabled(False)
            self.butconnect()
            self.canvas.setUpdatesEnabled(True)
        else:
            print("Point Type Not Supported")

    def delete_last_label(self):
        if not self.lane_labels or not self.lane_curve_artists:
            return
        last_curve = self.lane_curve_artists.pop()
        last_curve.remove()
        start, end = self.lane_labels.pop()
        try:
            for i in range(end-1, start-1, -1):
                artist = self.all_point_artists.pop(i)
                artist.remove()
        except Exception as e:
            print(f"[Delete Last Label] Error: {e}")
        if hasattr(self, 'lane_vtk_actors') and self.lane_vtk_actors:
            last_actor = self.lane_vtk_actors.pop()
            self.vtkRenderer.RemoveActor(last_actor)
            self.vtkWidget.GetRenderWindow().Render()
        self.canvas.draw()

    def delete_last_point(self):
        if not self.lane_point_artists or not self.lane_points:
            return
        last_artist = self.lane_point_artists.pop()
        last_artist.remove()
        self.lane_points.pop()
        self.canvas.draw()

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