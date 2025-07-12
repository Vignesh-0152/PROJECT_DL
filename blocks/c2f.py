import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.regularizers import l2
from .conv2d import conv2d
from .Bottleneck import Bottleneck
from .CspWithBottleneckWithSE import CspWithBottleneckWithSE
from .BottleneckWithSE import BottleneckWithSE

class c2f(Layer):
    """
        Cross-Stage Partial with Mixed Bottleneck Variants (C2F):
            This layer implements a hybrid feature-processing block inspired by CSPNet 
            and the C2f module seen in YOLOv8, combining multiple feature transformation 
            strategies for efficient representation learning.

            The block first applies a 1x1 convolution to unify the input channels. 
            Then it splits the feature map into four equal channel segments:
            - The first segment is passed through a standard Bottleneck block.
            - The second is processed using a custom CSP-style block with Squeeze-and-Excitation (SE).
            - The third segment is sent through a Bottleneck block that includes an SE module.
            - The fourth segment is left untouched, acting as a skip path.

            Finally, all processed segments are concatenated along the channel dimension, 
            enriching the output with a blend of residual, attention-based, and untouched signals.

        Args:
            filters (int, optional): Number of filters for the initial 1x1 convolution. Defaults to 256.
            splitNumber (int, optional): Number of channel splits. Currently Defaults to 4.
        
        Input:
            A 4D tensor of shape (batch_size, height, width, channels)

        Output:
            A 4D tensor of shape (batch_size, height, width, new_channels), where 
            new_channels = sum of output channels from each sub-block (typically same as input)

        Example:
            >>> layer = c2f(filters=256)
            >>> y = layer(x)  # x: input tensor
    """
    
    def __init__(
                    self, 
                    filters = None, 
                    splitNumber = 4
                ):
        
        super().__init__()
        self.filters = filters
        self.splitNumber = splitNumber

    def build(self, input_shape):
        self.conv2d = conv2d(
            filters = self.filters,
            kernel_size = (1,1),
            strides = 2
        )
        self.bottleneck = Bottleneck(64, 2)
        self.cspwithbottleneckwithse = CspWithBottleneckWithSE()
        self.bottleneckwithse = BottleneckWithSE(64, 64, 2, 4)

    def call(self, input):
        input = tf.cast(input, dtype= tf.float32)
        x = self.conv2d(input)
        x1, x2, x3, x4 = tf.split(x, num_or_size_splits= self.splitNumber, axis= -1)
        x1 = self.bottleneck(x1)
        x2 = self.cspwithbottleneckwithse(x2)
        x3 = self.bottleneckwithse(x3)
        
        x = tf.concat([x1,x2,x3,x4], axis= -1)
        return x
