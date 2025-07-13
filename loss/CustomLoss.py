import tensorflow as tf
from .bbloss import bbloss
from tensorflow.keras.losses import Loss
from .classloss import classloss
from .objloss import objloss

class CustomLoss(Loss):
    def __init__(self):
        super().__init__()
        self.bb_loss = bbloss()
        self.class_loss = classloss() 
        self.ob_loss = objloss()

    def call(self, y_true, y_pred):
        # bounding box loss
        bb_loss = self.bb_loss(y_true, y_pred)

        # class loss
        cls_loss = self.class_loss(y_true, y_pred)

        # object loss
        ob_loss = self.ob_loss(y_true, y_pred)

        total_loss = (0.5 * bb_loss) + (0.1 * cls_loss) + (0.1 * ob_loss)
        total_loss = tf.cast(total_loss, tf.float32)

        return total_loss