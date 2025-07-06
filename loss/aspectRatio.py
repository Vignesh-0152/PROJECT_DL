from .IoU import IoU
import tensorflow as tf
from numpy import pi

class aspectRatio:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred

    def __call__(self, *args, **kwds):
        return self.calculate()

    def calculate(self):
        cx1, cy1, w1, h1 = self.y_true[..., :4]
        cx2, cy2, w2, h2 = self.y_pred[..., :4]

        const = 4.0 / tf.square(pi)

        v = const * tf.square(tf.math.atan(w1 / (h1 + 1e-7)) - tf.math.atan(w2 / (h2 + 1e-7)))

        iou = IoU(self.y_true, self.y_pred)()
        divisor = (1 - iou) + v + 1e-7

        alpha = v / divisor

        aspectratio = v * alpha

        return aspectratio