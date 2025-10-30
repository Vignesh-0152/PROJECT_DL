from tensorflow.keras.layers import Layer
from .backbone import backbone
from .neck import neck
from .head import head
from utils import flattenandconcatenate
import tensorflow as tf

class CNN(Layer):
    def __init__(self, clss):
        super().__init__()
        self.clss = clss

    def build(self, input_shape):
        self.backbone = backbone()
        self.neck = neck()
        self.head = head(self.clss)
        self.flatandconcat = flattenandconcatenate()

    def call(self, input):
        c3, c4, c5 = self.backbone(input)
        p3, p4, p5 = self.neck(c3, c4, c5)
        h3, h4, h5 = self.head(p3, p4, p5)

        # Debug prints
        # tf.print("h3 shape:", tf.shape(h3))
        # tf.print("h4 shape:", tf.shape(h4))
        # tf.print("h5 shape:", tf.shape(h5))

        output = self.flatandconcat(h3, h4, h5)
        return output