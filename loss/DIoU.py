from utils import xycalc
import tensorflow as tf

class DIoU:
    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.DIoU_penalty = self.call()

    def __call__(self, *args, **kwds):
        return self.calculate()

    def __call__(self, *args, **kwds):
        return self.calculate()

    def calculate(self):
        cx1, cy1, w1, h1 = self.y_true[... , :4]
        cx2, cy2, w2, h2 = self.y_pred[... , :4]

        a = xycalc(cx1, cy1, w1, h1)
        b = xycalc(cx2, cy2, w2, h2)

        p2 = tf.square(cx1 - cx2) + tf.square(cy1 - cy2)

        c2 = tf.square(tf.maximum(a.x1, b.x1) - tf.minimum(a.x2, b.x2)) + tf.square(tf.maximum(a.y1, b.y1) - tf.minimum(a.y2, b.y2))

        DIoU_penalty = p2 / (c2 + 1e-7)

        return DIoU_penalty