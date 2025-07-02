import tensorflow as tf
from tensorflow.keras.layers import Layer,Conv2D,BatchNormalization,Activation
from tensorflow.keras.regularizers import l2

class Bottleneck(Layer):

    """
        Bottleneck block:
            This block compresses the number of channels do the computation and 
            again expands it back. It uses 3 Conv2D layers where first layer is 
            used for size reduction and last layer is used for restoring the 
            size. The middle layer is ued for feature extraction
        
        Logic:
            x = Conv2D( filter//y, (1x1))
            x = Conv2D( filter//y, (3x3))
            x = Conv2D( filter, (1x1))
        
        Args:
            filters(int): input the number of filters.( default: None)
            reductionRate(int): amount to which the filters are reduced in first two Conv2D layers.( default: 2 )

        Input:
            input the previous layer output of shape (x,x,filters)

        Output: 
            tensor of shape (x,x,filters)
    """

    def __init__(self, filters=None, reductionRate = 2):
        super().__init__()
        self.filters = filters
        self.reductionRate = reductionRate

    def build(self, input_shape):
        self.conv1 = Conv2D(
            int(self.filters // self.reductionRate),
            kernel_size = (1,1),
            strides = 1,
            padding = "same",
            kernel_regularizer = l2(0.01)
        )
        self.batchnorm1 = BatchNormalization(axis = -1)
        self.activation1 = Activation("swish")

        self.conv2 = Conv2D(
            int(self.filters // self.reductionRate),
            kernel_size = (3,3),
            strides = 1,
            padding = "same",
            kernel_regularizer = l2(0.01)
        )
        self.batchnorm2 = BatchNormalization(axis = -1)
        self.activation2 = Activation("swish")

        self.conv3 = Conv2D(
            self.filters,
            kernel_size = (1,1),
            strides = 1,
            padding = "same",
            kernel_regularizer = l2(0.01)
        )
        self.batchnorm3 = BatchNormalization(axis = -1)

    def call(self,input):
        x = self.conv1(input)
        x = self.batchnorm1(x)
        x = self.activation1(x)

        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)
        
        x = self.conv3(x)
        x = self.batchnorm3(x)

        return x
    