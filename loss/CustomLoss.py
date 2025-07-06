import tensorflow as tf
from .IoU import IoU
from .DIoU import DIoU
from .aspectRatio import aspectRatio
from tensorflow.keras.losses import Loss

class CustomLoss(Loss):
    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        iou = IoU(y_true, y_pred)()
        diou = DIoU(y_true, y_pred)()
        aspect_ratio_penalty = aspectRatio(y_true, y_pred)()

        loss = 1 - iou + diou + aspect_ratio_penalty

        return loss