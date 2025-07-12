from .Bottleneck import Bottleneck
from tensorflow.keras.layers import Layer,Activation
from tensorflow.keras.regularizers import l2
import tensorflow as tf

class ResNetWithBottleneck(Layer):
    """
        Residual Block with Bottleneck Architecture (ResNet-inspired):
            This layer applies a residual learning strategy around a Bottleneck block, 
            inspired by the original ResNet architecture. It enhances training stability 
            and feature reuse by introducing skip connections.

            The input is first passed through a custom Bottleneck layer. The output 
            is then added back to the original input (residual connection), promoting 
            gradient flow and preserving low-level features. A Swish (SiLU) activation 
            is applied at the end to introduce non-linearity.

            This design is useful in deep feature extractors where residual paths help 
            avoid vanishing gradients and support efficient convergence.

        Args:
            filters (int): Number of filters to configure within the block (currently unused, placeholder for extension).

        Input:
            A 4D tensor of shape (batch_size, height, width, channels)

        Output:
            A 4D tensor of shape (batch_size, height, width, channels), 
            same as the input due to the residual addition.

        Example:
            >>> layer = ResNetWithBottleneck(filters=64)
            >>> y = layer(x)  # x: input tensor
    """


    def __init__(self, filters):
        super().__init__()
        self.filters = filters

    def build(self, input_shape):
        self.bottleneck = Bottleneck(self.filters)
        self.activation1 = Activation("swish")

    def call(self, input):
        input = tf.cast(input, dtype= tf.float32)
        x = self.bottleneck(input)
        x = tf.add(input, x)
        x = self.activation1(x)

        return x