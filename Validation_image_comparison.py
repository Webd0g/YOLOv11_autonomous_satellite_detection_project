# Generated with chatgpt
from ultralytics import YOLO
import cv2, os
import matplotlib.pyplot as plt

model = YOLO(r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Trained_models\Final_models\Satellite_streaks_detection_model_14_Sept_2025\weights\best.pt")

# Pick some validation images
val_imgs = [
    r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Training_data_streaks\dataset\val\images\00077440_Az_26_Elev_42_5.000secs_Light.png" ,
    r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Training_data_streaks\dataset\val\images\00077447_Az_26_Elev_42_5.000secs_Light.png"
]

for img_path in val_imgs:
    # --- Ground truth ---
    img_gt = cv2.imread(img_path)
    h, w, _ = img_gt.shape

    label_path = img_path.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
    label_path = os.path.splitext(label_path)[0] + ".txt"

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.split())
                # convert YOLO format to pixel coords
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)
                cv2.rectangle(img_gt, (x1, y1), (x2, y2), (0,255,0), 2)

    # --- Model predictions ---
    results = model(img_path)[0]
    img_pred = results.plot()

    # --- Show comparison ---
    fig, axs = plt.subplots(1, 2, figsize=(12,6))
    axs[0].imshow(cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB))
    axs[0].axis("off")

    axs[1].imshow(cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB))
    axs[1].axis("off")

    plt.show()