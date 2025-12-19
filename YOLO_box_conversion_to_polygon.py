#source: generated with chatgpt
import os

def convert_bbox_to_polygon(bbox_line, img_w, img_h):
    """
    Convert a YOLO bbox line into a YOLO-seg polygon line (4 points rectangle).
    
    Args:
        bbox_line (str): YOLO bbox annotation line -> "<class> x_center y_center w h"
        img_w (int): image width
        img_h (int): image height

    Returns:
        str: YOLO-seg polygon line -> "<class> x1 y1 x2 y2 x3 y3 x4 y4"
    """
    parts = bbox_line.strip().split()
    cls = parts[0]
    x_c, y_c, w, h = map(float, parts[1:])

    # Convert normalized center format to pixel coordinates
    x_c *= img_w
    y_c *= img_h
    w *= img_w
    h *= img_h

    xmin = x_c - w / 2
    xmax = x_c + w / 2
    ymin = y_c - h / 2
    ymax = y_c + h / 2

    # Polygon (rectangle: top-left → top-right → bottom-right → bottom-left)
    x1, y1 = xmin / img_w, ymin / img_h
    x2, y2 = xmax / img_w, ymin / img_h
    x3, y3 = xmax / img_w, ymax / img_h
    x4, y4 = xmin / img_w, ymax / img_h

    polygon = f"{cls} {x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {x3:.6f} {y3:.6f} {x4:.6f} {y4:.6f}"
    return polygon


def convert_yolo_bbox_dir_to_seg(bbox_dir, seg_dir, img_w, img_h):
    """
    Convert all YOLO bbox labels in a directory into YOLO-seg polygon labels.

    Args:
        bbox_dir (str): directory with YOLO bbox labels (.txt)
        seg_dir (str): output directory for YOLO-seg polygon labels (.txt)
        img_w (int): image width
        img_h (int): image height
    """
    os.makedirs(seg_dir, exist_ok=True)

    for file in os.listdir(bbox_dir):
        if not file.endswith(".txt"):
            continue

        bbox_path = os.path.join(bbox_dir, file)
        seg_path = os.path.join(seg_dir, file)

        with open(bbox_path, "r") as f:
            lines = f.readlines()

        seg_lines = []
        for line in lines:
            seg_lines.append(convert_bbox_to_polygon(line, img_w, img_h))

        with open(seg_path, "w") as f:
            f.write("\n".join(seg_lines))

    print(f"✅ Conversion complete! Polygon labels saved to {seg_dir}")

convert_yolo_bbox_dir_to_seg(r'C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Training_data_streaks\dataset\val\labels',r'C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Training_data_segmented_streaks\dataset\val\labels', 461, 369)