"""Quick visualizer for all lane label JSONs.

Usage:
    python visualization.py

Iterates over every JSON in ./data/label/image and shows the image with lane
annotations.  Press any key (or close the window) to advance to the next
sample.  Press Ctrl+C to stop.
"""

import glob
import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np

# Folder containing label JSONs
LABEL_DIR = "./data/label/"

color_map = {
    "yellow_lane": (255, 255, 0),
    "white_lane": (255, 255, 255),
    "white_dash_lane": (180, 180, 180),
}

def visualize_sample(json_path: str):
    with open(json_path, "r") as f:
        data = json.load(f)

    img_path = data["file_path"]
    img = cv2.imread(img_path)
    if img is None:
        print(f"[WARN] Image not found: {img_path}")
        return
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.clf()
    plt.imshow(img_rgb)
    shown_labels = set()

    for lane in data.get("lane_lines", []):
        uv = np.asarray(lane.get("uv", []))
        if uv.size == 0:
            continue
        cat = lane.get("category", "Unknown")
        color = np.asarray(color_map.get(cat, (0, 255, 0))) / 255.0
        label = cat if cat not in shown_labels else None
        plt.scatter(uv[:, 0], uv[:, 1], color=color, s=50, label=label,
                    edgecolors="black", linewidths=1)
        shown_labels.add(cat)

    plt.title(os.path.basename(json_path))
    plt.axis("off")
    if shown_labels:
        plt.legend()
    plt.tight_layout()
    plt.draw()


def main():
    json_files = sorted(glob.glob(os.path.join(LABEL_DIR, "*.json")))
    if not json_files:
        print(f"No JSON files found in {LABEL_DIR}")
        return

    plt.figure(figsize=(16, 9))
    for jp in json_files:
        visualize_sample(jp)
        print(f"Showing {jp}. Close figure or press any key to continue …")
        # Wait until a key press or figure closed
        plt.waitforbuttonpress()
        if not plt.fignum_exists(plt.gcf().number):  # Figure closed
            break

if __name__ == "__main__":
    main()