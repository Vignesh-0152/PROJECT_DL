from preprocess import imgprocess

dataset = imgprocess(
        r"C:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\supervised_accidents_only\test\images", 
        r"c:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\supervised_accidents_only\test\labels"
    )
print((dataset))
