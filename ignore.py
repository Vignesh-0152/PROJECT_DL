from preprocess import imgprocess
import tensorflow as tf
import numpy as np
datasets = imgprocess(
        r"C:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\dataset-vehicles\valid\images", 
        r"C:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\dataset-vehicles\valid\labels"
    )
