import tensorflow as tf
from .IoU import IoU
from .DIoU import DIoU
from .aspectRatio import aspectRatio
from tensorflow.keras.losses import Loss

class bbloss:
    def __init__(self):
        super().__init__()

    def __call__(self, y_true, y_pred):
        return self.calculate(y_true= y_true, y_pred= y_pred)

    def calculate(self, y_true, y_pred):
        iou = IoU(y_true, y_pred)()
        diou = DIoU(y_true, y_pred)()
        aspect_ratio_penalty = aspectRatio(y_true, y_pred)()

        bb_loss = 1.0 - iou + diou + aspect_ratio_penalty
        bb_loss = tf.cast(bb_loss, tf.float32)

        bb_loss = tf.reduce_mean(bb_loss)

        return bb_loss