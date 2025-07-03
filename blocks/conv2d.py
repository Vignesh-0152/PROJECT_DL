import tensorflow as tf
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2

class conv2d(Layer):
    """    
        Conv2D layer:

        This block:
        - Applies Conv2D layer to the input with l2 regularization
        - Then BatchNormalization is appied
        - Activation of 'swish' is applied and output is returned

        Args:
            filters(int): It holds the number of channels this layer should output( default: 64 )
            kernel_size(tuple): The kernel_size for Conv2D.( default: (1,1) )
            strides(int): The stride that Conv2d kernel should move.( default: 1)
    """

    def __init__(self, filters= 64, kernel_size = (1,1), strides = 1):
        super().__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides

    def build(self, input_shape):
        self.conv2d = Conv2D(
            self.filters,
            self.kernel_size,
            strides = self.strides,
            padding = "same",
            kernel_regularizer = l2(0.01)            
        )
        self.batchnorm = BatchNormalization(axis = -1)
        self.activation = Activation("swish")

    def call(self, input):
        x = self.conv2d(input)
        x = self.batchnorm(x)
        x = self.activation(x)

        return x