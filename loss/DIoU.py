from utils import xycalc
import tensorflow as tf
from tensorflow.keras.losses import Loss

class DIoU(Loss):
    def __init__(self, y_true, y_pred):
        super().__init__()
        self.y_true = y_true
        self.y_pred = y_pred

    def __call__(self):
        return self.calculate()

    def calculate(self):
        # cx1, cy1, w1, h1 = self.y_true[... , 0:4]
        true_box = self.y_true[... , :4]
        cx1 = true_box[... , 0]
        cy1 = true_box[... , 1]
        w1 = true_box[... , 2]
        h1 = true_box[... , 3]
        # cx2, cy2, w2, h2 = self.y_pred[... , 0:4]
        pred_box = self.y_pred[... , 0:4]
        cx2 = pred_box[... , 0]
        cy2 = pred_box[... , 1]
        w2= pred_box[... , 2]
        h2 = pred_box[... , 3]

        a = xycalc(cx1, cy1, w1, h1)
        b = xycalc(cx2, cy2, w2, h2)

        p2 = tf.square(cx1 - cx2) + tf.square(cy1 - cy2)

        c2 = tf.square(tf.maximum(a.x1, b.x1) - tf.minimum(a.x2, b.x2)) + tf.square(tf.maximum(a.y1, b.y1) - tf.minimum(a.y2, b.y2))

        DIoU_penalty = p2 / (c2 + 1e-7)

        return DIoU_penalty