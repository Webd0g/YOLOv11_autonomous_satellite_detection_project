#Source: https://docs.ultralytics.com/modes/train/#usage-examples
from ultralytics import YOLO

#Load the model.
model = YOLO(r'C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Trained_models\Detection_models_4_09_2025\streak_detection_4th_sept_2025\weights\best.pt')

#Debug training run with augmentations.
results = model.train(data=r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Training_data_segmented_streaks\data.yaml",epochs=1, imgsz = 640, box = 8.0, cls = .3, dfl = 1.5, iou = 0.35, optimizer = 'AdamW', lr0 = 0.002, lrf = .1, cos_lr = True, hsv_h = 0.02, hsv_s = 0.4, hsv_v = 0.4, degrees = 10.0, translate = 0.04, scale = 0.1, save = True, save_period = -1, plots = True, val = True, project = r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Trained_models\10_09_2025", name = "Detection_Streaks_with_augments_debug_run")

#Debug training run without augmentations.
results = model.train(data=r'C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\September_04_05_training_and_validation_data\data.yaml',epochs=1, imgsz = 640, save = True, plots = True, val = True, project = r"C:\Users\riley\OneDrive\Documents\University\Year_3\Semester_2\Applied_science_project_2\machine_learning_detection_approach\Trained_models\14_09_2025", name = "14_09_2025_Detection_Streaks_debug_run")