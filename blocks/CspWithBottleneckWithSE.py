import tensorflow as tf
from tensorflow.keras.layers import Layer
from .BottleneckWithSE import BottleneckWithSE
from .conv2d import conv2d

class CspWithBottleneckWithSE(Layer):
    """
        Cross Stage Partial Block with Bottleneck + Squeeze-and-Excitation (CSP-SE):
            This layer implements a CSP (Cross Stage Partial) strategy that splits the 
            input feature map into two halves along the channel axis. One half is passed 
            through a Bottleneck block augmented with a Squeeze-and-Excitation (SE) 
            module, while the other half is kept as-is to preserve gradient flow and 
            original context.

            After processing, both halves are concatenated along the channel dimension 
            and passed through a 3x3 convolution to merge the features. This structure 
            helps to balance parameter efficiency and gradient propagation while improving 
            the channel attention and representational power.

        Args:
            filters (int, optional): Number of filters for the output convolution. If None, it uses the same as the input channel size.
            splitNumber (int, optional): Controls how the input channels are split. Default is 2 (splits into two halves).

        Input:
            A 4D tensor of shape (batch_size, height, width, channels)

        Output:
            A 4D tensor of shape (batch_size, height, width, channels), 
            where channels = original input channels (unless filters is manually overridden)

        Example:
            >>> layer = CspWithBottleneckWithSE()
            >>> y = layer(x)  # x: input tensor
    """

    def __init__(self, filters = None, splitNumber = 2):
        super().__init__()
        self.filters = filters
        self.splitNumber = splitNumber
    
    def build(self, input_shape):
        num_channels = input_shape[-1]
        self.bottleneckwithse = BottleneckWithSE(
                num_channels//self.splitNumber , 
                num_channels//self.splitNumber , 
                2, 
                4
            )
        self.conv2d = conv2d(
                num_channels , 
                (3,3), 
                1
            )

    def call(self, input):
        input = tf.cast(input, dtype= tf.float32)
        x1, x2 = tf.split(input, num_or_size_splits = 2, axis = -1)
        x = self.bottleneckwithse(x2)
        x = tf.concat([x1,x], axis = -1)
        x = self.conv2d(x)

        return x
