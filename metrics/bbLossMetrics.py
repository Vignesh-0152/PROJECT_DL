from tensorflow.keras.metrics import Metric
import tensorflow as tf
from loss import bbloss

class bbLossMetrics(Metric):
    def __init__(self, name= 'bbox_loss'):
        super().__init__(name= name)
        self.total = self.add_weight(name= 'total', initializer= 'zeros')
        self.count = self.add_weight(name= 'count', initializer= 'zeros')
        self.bbloss = bbloss()

    def update_state(self, y_true, y_pred):
        bb_loss = self.bbloss(y_true, y_pred)
        self.total.assign_add(bb_loss)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + 1e-7)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)