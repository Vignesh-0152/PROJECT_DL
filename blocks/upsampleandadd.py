import tensorflow as tf
from tensorflow.keras.layers import Layer, UpSampling2D
from tensorflow.keras.regularizers import l2
from .conv2d import conv2d

class upsampleandadd(Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_size):
        self.conv2d = conv2d(filters = 256, kernel_size = (1,1), strides = 1)
        self.upsample = UpSampling2D()

    def call(self, cx, py):
        py_up = self.upsample(py)
        cx_conv2d = self.conv2d(cx)
        py_out = tf.add(py_up, cx_conv2d)

        return py_out