from tensorflow.keras.metrics import Metric
import tensorflow as tf
from utils import xycalc

class IoUMetrics(Metric):
    def __init__(self, name= 'IoU_Metrics'):
        super().__init__(name= name)
        self.total = self.add_weight(name= 'total', initializer= 'zeros')
        self.count = self.add_weight(name= 'count', initializer= 'zeros')

    def update_state(self, y_true, y_pred):
        true_box = y_true[... , :4]
        cx1 = true_box[... , 0]
        cy1 = true_box[... , 1]
        w1 = true_box[... , 2]
        h1 = true_box[... , 3]
        # cx2, cy2, w2, h2 = self.y_pred[... , 0:4]
        pred_box = y_pred[... , 0:4]
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

        IoU_Metrics = intersection / (union + 1e-7)
        IoU_Metrics = tf.reduce_mean(IoU_Metrics)

        self.total.assign_add(IoU_Metrics)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + 1e-7)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)