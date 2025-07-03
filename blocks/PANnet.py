import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from .conv2d import conv2d
from .downandconcat import downandconcat

class PANet(Layer):

    def __init__(self):
        super().__init__()

    def build(self, input_shape):
        self.p3_p4_downandconcat = downandconcat()
        self.p4_p5_downandconcat = downandconcat()
        self.convp3 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.convp4 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.convp5 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.drop3 = Dropout(0.05)
        self.drop4 = Dropout(0.05)
        self.drop5 = Dropout(0.05)

    def call(self, p3, p4, p5):
        p4_pan_out = self.p3_p4_downandconcat(p3, p4)
        p5_pan_out = self.p4_p5_downandconcat(p4_pan_out, p5)
        
        p3_pan_out = self.convp3(p3)
        p4_pan_out = self.convp4(p4_pan_out)
        p5_pan_out = self.convp5(p5_pan_out)

        p3_pan_out = self.drop3(p3_pan_out)
        p4_pan_out = self.drop4(p4_pan_out)
        p5_pan_out = self.drop5(p5_pan_out)

        return p3_pan_out, p4_pan_out, p5_pan_out