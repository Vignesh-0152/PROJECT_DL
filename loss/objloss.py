import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.losses import Loss

class objloss(Loss):
    def __init__(self):
        super().__init__()
        self.bce = BinaryCrossentropy(from_logits= False, reduction='sum_over_batch_size')

    def __call__(self, y_true, y_pred):
        return self.calculate(y_true= y_true, y_pred= y_pred)

    def calculate(self, y_true, y_pred):
        ob_y_true = y_true[... , 4]
        ob_y_pred = y_pred[... , 4]

        weights = ob_y_true * 3.0 + 1.0    # (B, 8400)
        weights = tf.expand_dims(weights, axis=-1)  # Now (B, 8400, 1)

        ob_loss = self.bce(ob_y_true, ob_y_pred, sample_weight= weights)
        ob_loss = tf.cast(ob_loss, tf.float32)

        return ob_loss