import os
import cv2
import sys
import vtk
import numpy as np
import open3d as o3d

from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap, QKeySequence
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QShortcut

from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

class PreCalibrationUI(QWidget):
    def __init__(self, img_dir, pcd_dir, cx, cy, cz, dx, dy, dz):
        super().__init__()
        self.setWindowTitle("Crop ROI for Calibration")
        self.resize(1200, 600)
        self.setFocusPolicy(Qt.StrongFocus)  # Set focus policy early
        
        # Store directories
        self.img_dir = img_dir
        self.pcd_dir = pcd_dir
        
        # Create output directories if they don't exist
        os.makedirs('./data/RoiPCD/', exist_ok=True)
        os.makedirs('./data/WholePCD/', exist_ok=True)
        os.makedirs('./data/Image/', exist_ok=True)
        
        # Get sorted lists of image and pcd files
        self.img_files = sorted([f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
        self.pcd_files = sorted([f for f in os.listdir(pcd_dir) if f.endswith('.pcd')])
        
        # Current file index
        self.current_idx = 0
        
        # Validate that we have matching files
        if not self.img_files or not self.pcd_files:
            raise FileNotFoundError("No image or PCD files found in the specified directories")
            
        # Get initial file paths
        self.current_img_path = os.path.join(img_dir, self.img_files[0])
        self.current_pcd_path = os.path.join(pcd_dir, self.pcd_files[0])
        # Set navy background
        # self.setStyleSheet("""
        #     QWidget, QWidget * {
        #         background-color: #001f3f;
        #         color: white;
        #     }
        #     QLabel, QLineEdit, QPushButton {
        #         color: white;
        #     }
        #     QLineEdit {
        #         border: 1px solid white;
        #         color: white;
        #         selection-background-color: #004080;
        #         selection-color: white;
        #         caret-color: white;
        #     }
        #     QLineEdit::placeholder {
        #         color: white;
        #     }
        #     QPushButton {
        #         border: 1px solid white;
        #         background-color: #002b5c;
        #         color: white;
        #     }
        # """)

        # size location
        self.cx = cx
        self.cy = cy
        self.cz = cz
        self.dx = dx
        self.dy = dy
        self.dz = dz

        # Layouts
        layout = QHBoxLayout(self)
        left_layout = QVBoxLayout()   # For image
        center_layout = QVBoxLayout() # For VTK
        right_layout = QVBoxLayout()  # For input fields

        # --- Image (matplotlib) ---
        self.figure = Figure(figsize=(6,4))
        self.canvas = FigureCanvas(self.figure)
        self.ax = self.figure.add_subplot(111)
        left_layout.addWidget(self.canvas)
        self.load_and_display_image()

        # --- VTK PCD Viewer ---
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        self.vtk_renderer = vtk.vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.vtk_renderer)
        center_layout.addWidget(self.vtk_widget)

        # --- Box Widget for 3D ROI (must be before plot_pointcloud) ---
        self.box_widget = vtk.vtkBoxWidget()
        self.box_widget.SetInteractor(self.vtk_widget.GetRenderWindow().GetInteractor())
        self.box_widget.SetPlaceFactor(1.0)  # Use exact size
        self.box_widget.TranslationEnabledOn()
        self.box_widget.ScalingEnabledOff()
        self.box_widget.RotationEnabledOff()
        self.box_widget.AddObserver('InteractionEvent', self.on_box_interaction)
        self._box_bounds = None  # Store last box bounds
        self._all_points = None  # Store all points for cropping
        self._all_colors = None
        self._last_polydata = None


        right_layout.addStretch(1)

        # --- Navigation Buttons ---
        btn_row = QHBoxLayout()
        self.prev_btn = QPushButton("Prev")
        self.next_btn = QPushButton("Next")
        btn_row.addWidget(self.prev_btn)
        btn_row.addWidget(self.next_btn)
        right_layout.addLayout(btn_row)

        # --- Apply Button ---
        self.crop_btn = QPushButton('Apply Crop')
        right_layout.addWidget(self.crop_btn)
        self.crop_btn.clicked.connect(self.on_apply_clicked)
        self._box_widget_enabled = False
        
        # Connect prev/next buttons after they are initialized
        self.prev_btn.clicked.connect(self.prev_file)
        self.next_btn.clicked.connect(self.next_file)
        
        # Update button states
        self.update_button_states()
        
        # --- Now safe to plot (box_widget exists) ---
        # Ensure these are NOT reset after plot_pointcloud!
        self.load_pointcloud()

        # --- Combine layouts ---
        layout.addLayout(left_layout, 2)
        layout.addLayout(center_layout, 3)
        layout.addLayout(right_layout, 1)
        self.setLayout(layout)
        
        # Create a shortcut for Apply button
        self.shortcut_apply = QShortcut(QKeySequence('s'), self)
        self.shortcut_apply.activated.connect(self.on_apply_clicked)
        
        # Create a shortcut for Previous button
        self.shortcut_prev = QShortcut(QKeySequence('a'), self)
        self.shortcut_prev.activated.connect(self._on_prev_shortcut)
        
        # Create a shortcut for Next button
        self.shortcut_next = QShortcut(QKeySequence('d'), self)
        self.shortcut_next.activated.connect(self._on_next_shortcut)
        
    def _on_prev_shortcut(self):
        if self.prev_btn.isEnabled():
            self.prev_file()
            
    def _on_next_shortcut(self):
        if self.next_btn.isEnabled():
            self.next_file()

    def plot_pointcloud(self, pcd_path, points=None, colors=None):
        self.vtk_renderer.RemoveAllViewProps()
        # Load from file if not provided
        if points is None:
            if not os.path.exists(pcd_path):
                print(f"PCD file not found: {pcd_path}")
                self._all_points = None
                self._all_colors = None
                self._last_polydata = None
                return
            pcd = o3d.t.io.read_point_cloud(pcd_path)
            if 'positions' not in pcd.point:
                print(f"No valid points in PCD: {pcd_path}")
                self._all_points = None
                self._all_colors = None
                self._last_polydata = None
                return
            np_points = pcd.point.positions.numpy()
            if np_points is None or np_points.shape[0] == 0:
                print("Loaded point cloud is empty!")
                self._all_points = None
                self._all_colors = None
                self._last_polydata = None
                return
                
            # Convert to float32 if needed and check for invalid values
            np_points = np.asarray(np_points, dtype=np.float32)
            mask_valid = np.isfinite(np_points).all(axis=1)
            
            if not np.all(mask_valid):
                print(f"[plot_pointcloud] Warning: Removing {np.sum(~mask_valid)} points with NaN/inf!")
                np_points = np_points[mask_valid]
            
            self._all_points = np_points
            # Process colors if points are valid
            if mask_valid.any():
                if 'intensity' in pcd.point:
                    print("intensity")
                    np_intensity = pcd.point.intensity.numpy().flatten()
                    if len(np_intensity) > len(mask_valid):  # In case intensity has different length
                        np_intensity = np_intensity[:len(mask_valid)]
                    if len(np_intensity) > 0:
                        np_intensity = np_intensity[mask_valid]
                        min_i, max_i = np_intensity.min(), np_intensity.max()
                        if max_i > min_i:
                            norm_i = ((np_intensity - min_i) / (max_i - min_i) * 255).astype(np.uint8)
                        else:
                            norm_i = np.zeros_like(np_intensity, dtype=np.uint8)
                        np_colors = np.stack([norm_i]*3, axis=1)
                    else:
                        np_colors = np.zeros((len(np_points), 3), dtype=np.uint8)
                elif 'colors' in pcd.point:
                    print("colors")
                    colors = pcd.point.colors.numpy()
                    if len(colors) > len(mask_valid):  # In case colors has different length
                        colors = colors[:len(mask_valid)]
                    if len(colors) > 0:
                        np_colors = (colors[mask_valid] * 255).astype(np.uint8)
                    else:
                        np_colors = np.zeros((len(np_points), 3), dtype=np.uint8)
                else:
                    print("gray")
                    np_colors = np.full((len(np_points), 3), 128, dtype=np.uint8)
                
                self._all_colors = np_colors
            else:
                print("No valid points to display colors")
                self._all_colors = None
        else:
            np_points = points
            np_colors = colors
            self._all_points = np_points
            self._all_colors = np_colors
        if self._all_points is None or self._all_colors is None or self._all_points.shape[0] == 0:
            print("[plot_pointcloud] No valid points/colors to display!")
            self._last_polydata = None
            return
        # VTK points/colors
        vtk_points = vtk.vtkPoints()
        vtk_colors = vtk.vtkUnsignedCharArray()
        vtk_colors.SetNumberOfComponents(3)
        vtk_colors.SetName("Colors")
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
        self.vtk_renderer.AddActor(actor)
        self.vtk_renderer.SetBackground(1,1,1)
        self.vtk_renderer.ResetCamera()
        camera = self.vtk_renderer.GetActiveCamera()
        camera.SetPosition(5, 0, 30)  # 더 높은 위치(Z=30)에서
        camera.SetFocalPoint(5, 0, 0)  # 아래(지면) 방향으로 바라봄
        camera.SetViewUp(0, 1, 0)      # 위쪽이 Y축
        self.vtk_renderer.ResetCameraClippingRange()
        style = vtk.vtkInteractorStyleTrackballCamera()
        style.SetMotionFactor(4)
        self.vtk_widget.GetRenderWindow().GetInteractor().SetInteractorStyle(style)
        self.vtk_widget.GetRenderWindow().Render()
        # Place box widget around points
        bounds = polydata.GetBounds()
        self._last_polydata = polydata
        self._all_points = np_points
        self._all_colors = np_colors
        self.box_widget.SetInputData(polydata)
        # Shrink box to 20% of bounding box and center at ego (origin or center)
        # roi_bounds = [self.cx - self.dx/2, self.cx + self.dx/2, self.cy - self.dy/2, self.cy + self.dy/2, self.cz - self.dz/2, self.cz + self.dz/2]
        # self.box_widget.PlaceWidget(roi_bounds)
        # Set box outline color to red for visibility
        try:
            self.box_widget.GetOutlineProperty().SetColor(1, 0, 0)  # Red color
            self.box_widget.GetOutlineProperty().SetLineWidth(2.0)  # Make the outline more visible
        except AttributeError:
            # fallback for VTK versions without GetOutlineProperty
            try:
                self.box_widget.GetHandleProperty().SetColor(1, 0, 0)
            except Exception:
                pass
        
        # Enable and place the box widget
        roi_bounds = [
            self.cx - self.dx/2, self.cx + self.dx/2,
            self.cy - self.dy/2, self.cy + self.dy/2,
            self.cz - self.dz/2, self.cz + self.dz/2
        ]
        self.box_widget.PlaceWidget(roi_bounds)
        self.box_widget.EnabledOn()  # Enable the box widget
        self.vtk_widget.GetRenderWindow().Render()  # Force render update

    def on_box_interaction(self, caller, event):
        # Get the current transform from the box widget
        transform = vtk.vtkTransform()
        self.box_widget.GetTransform(transform)
        
        # Get the box widget's polydata
        polydata = vtk.vtkPolyData()
        self.box_widget.GetPolyData(polydata)
        
        # Apply the transform to get the actual bounds
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetTransform(transform)
        transform_filter.SetInputData(polydata)
        transform_filter.Update()
        
        # Get the transformed bounds
        bounds = transform_filter.GetOutput().GetBounds()
        min_x, max_x, min_y, max_y, min_z, max_z = bounds
        self._box_bounds = (min_x, max_x, min_y, max_y, min_z, max_z)
        
        # Calculate center point of the box
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2
        center_z = (min_z + max_z) / 2

        # Force render update
        self.vtk_widget.GetRenderWindow().Render()
        
    def on_apply_clicked(self):
        # Apply 버튼을 누르면 바로 크롭 실행
        if self._last_polydata is None or self._all_points is None or self._all_colors is None or self._all_points.shape[0] == 0:
            print('[on_apply_clicked] 포인트 클라우드가 없거나 비어 있습니다!')
            return
        
        # Update box bounds one last time before cropping
        self.on_box_interaction(self.box_widget, None)
        print(f'[Apply Clicked] Final box bounds: {self._box_bounds}')
            
        # 크롭 실행
        self.crop_roi()



    def crop_roi(self):
        # 1) BoxWidget에서 저장된 Bounds 가져오기
        if self._box_bounds:
            min_x, max_x, min_y, max_y, min_z, max_z = self._box_bounds
        else:
            # Get the transform from the box widget
            transform = vtk.vtkTransform()
            self.box_widget.GetTransform(transform)
            transform.Inverse()
            
            # Get the box widget's polydata
            polydata = vtk.vtkPolyData()
            self.box_widget.GetPolyData(polydata)
            
            # Transform the polydata to get the bounds
            transform_filter = vtk.vtkTransformPolyDataFilter()
            transform_filter.SetTransform(transform)
            transform_filter.SetInputData(polydata)
            transform_filter.Update()
            
            # Get the transformed bounds
            bounds = transform_filter.GetOutput().GetBounds()
            min_x, max_x, min_y, max_y, min_z, max_z = bounds

        box_center = np.array([(min_x + max_x)/2, (min_y + max_y)/2, (min_z + max_z)/2])
        spawn_center = np.array([self.cx, self.cy, self.cz])
        
        # Calculate offset (difference between box center and spawn center)
        offset = box_center - spawn_center
        
        # Double the offset and apply to the spawn center
        adjusted_center = spawn_center + offset / 2
        
        # Calculate new bounds around the adjusted center
        half_dx = self.dx / 2
        half_dy = self.dy / 2
        half_dz = self.dz / 2
        
        min_x = adjusted_center[0] - half_dx
        max_x = adjusted_center[0] + half_dx
        min_y = adjusted_center[1] - half_dy
        max_y = adjusted_center[1] + half_dy
        min_z = adjusted_center[2] - half_dz
        max_z = adjusted_center[2] + half_dz
        print(f"[Crop Info] Final crop bounds:")
        print(f"  X: [{min_x:.4f}, {max_x:.4f}]")
        print(f"  Y: [{min_y:.4f}, {max_y:.4f}]")
        print(f"  Z: [{min_z:.4f}, {max_z:.4f}]")
        pts = self._all_points
        if pts is None or pts.size == 0:
            print("[crop_roi] No points to crop!")
            return

        # 2) 단순 Axis-aligned bounding box 필터
        mask = (
            (pts[:, 0] >= min_x) & (pts[:, 0] <= max_x) &
            (pts[:, 1] >= min_y) & (pts[:, 1] <= max_y) &
            (pts[:, 2] >= min_z) & (pts[:, 2] <= max_z)
        )
        cropped_points = pts[mask]
        cropped_colors = None
        if self._all_colors is not None and len(self._all_colors) == len(pts):
            cropped_colors = self._all_colors[mask]

        if cropped_points.shape[0] == 0:
            print("[crop_roi] No points to crop!")
            return

        # 3) Save cropped data in NPY format
        # Create ROI_NPY directory if it doesn't exist
        roi_npy_dir = './data/RoIPCD/'
        os.makedirs(roi_npy_dir, exist_ok=True)
        
        # Get the original PCD filename and create output path for NPY
        original_filename = os.path.basename(self.current_pcd_path).replace('.pcd', '.npy')
        output_npy_path = os.path.join(roi_npy_dir, original_filename)
        
        # Save points as NPY
        np.save(output_npy_path, cropped_points)
        print(f"[crop_roi] Saved cropped ROI points to {output_npy_path}")

        # 4) Update internal state and redraw
        self._all_points = cropped_points
        self._all_colors = cropped_colors
        self.plot_pointcloud(None, cropped_points, cropped_colors)

        # 5) Disable box widget
        self.box_widget.EnabledOff()

    def load_and_display_image(self):
        """Load and display the current image"""
        img = cv2.imread(self.current_img_path)
        if img is None:
            print(f"[ERROR] Failed to load image: {self.current_img_path}")
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.ax.clear()
        self.ax.imshow(img)
        self.ax.axis('off')
        self.canvas.draw()
        self.setWindowTitle(f"Crop ROI for Calibration - {os.path.basename(self.current_img_path)}")
    
    def load_pointcloud(self):
        """Load and display the current point cloud"""
        # Save original point cloud as NPY before plotting
        if hasattr(self, 'current_pcd_path'):
            pcd = o3d.t.io.read_point_cloud(self.current_pcd_path)
            if 'positions' in pcd.point:
                np_points = pcd.point.positions.numpy()
                # Save original points to Whole_PCD folder
                output_npy_path = os.path.join('./data/WholePCD/', 
                                            os.path.basename(self.current_pcd_path).replace('.pcd', '.npy'))
                np.save(output_npy_path, np_points)
                print(f"[load_pointcloud] Saved original points to {output_npy_path}")
        
        self.plot_pointcloud(self.current_pcd_path)
    
    def update_button_states(self):
        """Update the enabled state of navigation buttons"""
        self.prev_btn.setEnabled(self.current_idx > 0)
        self.next_btn.setEnabled(self.current_idx < min(len(self.img_files), len(self.pcd_files)) - 1)
    
    def prev_file(self):
        """Load the previous file pair"""
        if self.current_idx > 0:
            self.current_idx -= 1
            self.load_file_pair()
    
    def next_file(self):
        """Load the next file pair"""
        if self.current_idx < min(len(self.img_files), len(self.pcd_files)) - 1:
            self.current_idx += 1
            self.load_file_pair()
    
    def load_file_pair(self):
        """Load the current image and point cloud pair"""
        self.current_img_path = os.path.join(self.img_dir, self.img_files[self.current_idx])
        self.current_pcd_path = os.path.join(self.pcd_dir, self.pcd_files[self.current_idx])
        
        self.load_and_display_image()
        self.load_pointcloud()
        self.update_button_states()

if __name__ == "__main__":
    # bbox center point
    cx = 10.0
    cy = 0.0
    cz = 1.5
    # bbox size
    dx = 2.5
    dy = 2.5
    dz = 2.5

    # Use directories instead of individual files
    img_dir = "./data_for_calib/image/"
    pcd_dir = "./data_for_calib/PCD/"
    
    app = QApplication(sys.argv)
    window = PreCalibrationUI(img_dir, pcd_dir, cx, cy, cz, dx, dy, dz)
    window.show()
    sys.exit(app.exec_())
