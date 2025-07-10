import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy

class classloss:
    def __init__(self):
        self.scce = SparseCategoricalCrossentropy(from_logits= False)

    def __call__(self, y_true, y_pred):
        return self.calculate(y_true= y_true, y_pred= y_pred)

    def calculate(self, y_true, y_pred):
        cls_y_true = y_true[... , 5:]
        cls_y_pred = y_pred[... , 5:]
        cls_loss = self.scce(cls_y_true, cls_y_pred)
        cls_loss = tf.cast(cls_loss, tf.float32)

        return cls_loss