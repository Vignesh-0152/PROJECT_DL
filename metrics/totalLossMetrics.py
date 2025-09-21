from tensorflow.keras.metrics import Metric
from loss import CustomLoss
import tensorflow as tf

class totalLossMetrics(Metric):
    def __init__(self, name= 'total_loss'):
        super().__init__(name= name)
        self.total = self.add_weight(name= 'total', initializer= 'zeros')
        self.count = self.add_weight(name= 'count', initializer= 'zeros')
        self.CustomLoss = CustomLoss()

    def update_state(self, y_true, y_pred):
        total_loss = self.CustomLoss(y_true, y_pred)
        self.total.assign_add(total_loss)
        self.count.assign_add(1.0)

    def result(self):
        return self.total / (self.count + 1e-7)

    def reset_states(self):
        self.total.assign(0.0)
        self.count.assign(0.0)