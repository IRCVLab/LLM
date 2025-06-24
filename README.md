<h1 align="center">3D Lane Labeling Machine</h1>
<p align="center">The 3D Lane Labeling Machine is a tool designed for autonomous driving applications, enabling efficient labeling of 3D lane data. This tool helps create training datasets for autonomous vehicle perception systems by providing an intuitive interface for labeling lane markings in 3D space.</p>
<!-- ![title.png](title.png) -->



## 🚀 Overview


Features include:
- 3D visualization of camera images and LiDAR point clouds
- Interactive lane point labeling
- Support for **OpenLane V1** data format
- Camera and LiDAR calibration support
- Easy-to-use graphical interface

## 🔨 Installation

```bash
pip install -r requirements.txt
```

## ▶️  Execution

```bash
python main.py
```

## 📄 Documentation
This version supports the **OpenLane V1** format and requires *images*, *PCD files*, and *camera and lidar calibration files*.

#### Data
- Put your images in `data/image/image` folder.
- Put your pcd files in `data/image/pcd` folder.
- Put your calibration files in `calibration` folder. (`r.txt`, `t.txt`, `k.txt`, `distortion.txt`)

#### Usage

1. Run `python main.py`
2. Choose the class of lane. 
3. Click the point you want to label on the left image or the right  window.
4. Click the 'Add Lane' button. Also, you can delete the points and lane by clicking the 'Delete' button.
5. Click the 'Save' button to save the label or 'Next' button to save and move to the next image. (`data/label `folder)

- You can use keyboard shortcuts to add, delete, move and so on..
---

This project is based on: [Road_Labeler](https://github.com/InhwanBae/Road_Labeler)