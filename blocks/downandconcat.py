import tensorflow as tf
from tensorflow.keras.layers import Layer
from .conv2d import conv2d

class downandconcat(Layer):
    """
        Downsample and Concatenate Layer (PANet Bottom-Up Block):
            Used in the PANet to aggregate shallow features with deeper ones.
            It downsamples the higher-resolution feature map (px) and 
            concatenates it with a lower-resolution map (py).
        
        Methods:
            - build: defines the Conv2D(1x1, stride=2) for downsampling px
            - call: applies downsampling to px and concatenates it with py

        Input:
            px: higher-resolution feature map, shape (B, H, W, C)
            py: lower-resolution feature map, shape (B, H/2, W/2, C)

        Output:
            Concatenated feature map, shape (B, H/2, W/2, C + 256)
    """
    def __init__(self):
        super().__init__()

    def build(self, input_size):
        self.down = conv2d(filters = 256, kernel_size = (1,1), strides = 2)
    
    def call(self, px, py):
        px_down = self.down(px)
        px_concat = tf.concat([py, px_down], axis = -1)

        return px_concat