import tensorflow as tf
from tensorflow.keras.layers import Layer, UpSampling2D
from .conv2d import conv2d

class upsampleandadd(Layer):
    """
        Upsample and Add Layer:
            This layer performs top-down feature fusion in Feature Pyramid Networks (FPN).
            It takes a deeper feature map and a shallower lateral feature map, upsamples 
            the deeper one to match the spatial size of the shallower, and adds them 
            element-wise after applying a 1x1 Conv2D on the shallower input to align channels.

        Logic:
            Step 1: py_up = UpSample(py)
            Step 2: cx_conv2d = Conv2D(1x1)(cx)
            Step 3: py_out = py_up + cx_conv2d

        Args:
            None

        Input:
            cx: lateral feature map from the backbone (higher resolution), shape = (B, H, W, C)
            py: top-down feature map from the FPN (lower resolution), shape = (B, H/2, W/2, 256)

        Output:
            py_out: fused feature map with shape = (B, H, W, 256)

        Notes:
            - Assumes the target output filter dimension is 256
            - The 1x1 Conv2D layer ensures cx has same number of channels as py
            - Uses nearest neighbor upsampling by default via UpSampling2D
            - Designed specifically for FPN top-down pathway
    """
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