import tensorflow as tf
from utils import xycalc
from tensorflow.keras.losses import Loss

class IoU(Loss):
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

        inter_x1 = tf.maximum(a.x1, b.x1)
        inter_y1 = tf.maximum(a.y1, b.y1)
        inter_x2 = tf.minimum(a.x2, b.x2)
        inter_y2 = tf.minimum(a.y2, b.y2)

        inter_iw = tf.maximum(0.0, inter_x2 - inter_x1)
        inter_ih = tf.maximum(0.0, inter_y2 - inter_y1)

        intersection = inter_iw * inter_ih
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)

        union = area_a + area_b - intersection

        iou = intersection / (union + 1e-7)

        return iou