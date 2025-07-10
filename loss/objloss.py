import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

class objloss:
    def __init__(self):
        self.bce = BinaryCrossentropy(from_logits= False)

    def __call__(self, y_true, y_pred):
        return self.calculate(y_true= y_true, y_pred= y_pred)

    def calculate(self, y_true, y_pred):
        ob_y_true = y_true[... , 4]
        ob_y_pred = y_pred[... , 4]
        ob_loss = self.bce(ob_y_true, ob_y_pred)
        ob_loss = tf.cast(ob_loss, tf.float32)

        return ob_loss