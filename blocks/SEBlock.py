import tensorflow as tf
from tensorflow.keras.layers import Layer,GlobalAveragePooling2D,Dense,BatchNormalization,Activation,Dropout
from tensorflow.keras.regularizers import l2

class SEBlock(Layer):
    """
        SEBlock (Squeeze-and-Excitation Block)

        This block performs channel-wise attention by:
        - Squeezing spatial info using GlobalAveragePooling2D
        - Learning channel importance via Dense layers with Swish + Sigmoid
        - Reweighting the input tensor via element-wise multiplication
        - Includes BatchNormalization, Dropout(0.05), and L2 regularization

        Args:
            filters (int): Number of input/output filters (default: None)
            reductionRate (int): Reduction ratio for bottleneck (default: 4)

        Returns:
            A tensor with the same shape as input, with recalibrated features
    """

    def __init__(self, filters = None, reductionRate = 4):
        super().__init__()
        self.filters = filters
        self.reductionRate = reductionRate 
    
    def build(self,input_shape):
        self.globalaveragepool = GlobalAveragePooling2D()

        self.dense1 = Dense(
            int(self.filters // self.reductionRate), 
            use_bias=True, 
            kernel_initializer="he_normal", 
            bias_initializer="zeros", 
            kernel_regularizer=l2(0.01) 
        )

        self.dense2 = Dense(
            self.filters,
            use_bias=True,
            kernel_initializer="he_normal",
            bias_initializer="zeros",
            kernel_regularizer=l2(0.01)
        )

        self.dropout = Dropout(0.05)
        self.batchnorm1 = BatchNormalization(axis = -1)
        self.batchnorm2 = BatchNormalization(axis = -1)
        self.activation1 = Activation("swish")
        self.activation2 = Activation("sigmoid")

    def call(self,input):
        x = self.globalaveragepool(input)
        x = self.dense1(x)
        x = self.batchnorm1(x)
        x = self.activation1(x)

        x = self.dense2(x)
        x = self.batchnorm2(x)
        x = self.activation2(x)

        x = tf.reshape(x, (-1,1,1,tf.shape(input)[-1]))
        x = tf.multiply(input,x)
        x = self.dropout(x)

        return x

