import tensorflow as tf
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.losses import Loss

class classloss(Loss):
    def __init__(self):
        super().__init__()
        self.scce = SparseCategoricalCrossentropy(from_logits=False, reduction='sum_over_batch_size')

    def call(self, y_true, y_pred):
        return self.calculate(y_true=y_true, y_pred=y_pred)

    @tf.autograph.experimental.do_not_convert
    def calculate(self, y_true, y_pred):
        # Adjust slicing if needed
        cls_y_true = y_true[..., 5]
        # If your model outputs more than 6 features, adjust slicing here
        cls_y_pred = y_pred[..., 5:]
        # Reshape to (batch_size * total_cells, num_classes)
        cls_y_pred = tf.reshape(cls_y_pred, [-1, tf.shape(cls_y_pred)[-1]])
        cls_y_true = tf.reshape(cls_y_true, [-1])
        cls_loss = self.scce(cls_y_true, cls_y_pred)
        cls_loss = tf.cast(cls_loss, tf.float32)
        return cls_loss