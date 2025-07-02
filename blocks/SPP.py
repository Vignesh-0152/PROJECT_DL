import tensorflow as tf
from tensorflow.keras.layers import Layer,MaxPool2D
from tensorflow.keras.regularizers import l2

class SPP(Layer):
    """
        Spatial Pyramid Pooling (SPP) Layer:
            The SPP block captures multi-scale spatial context by applying 
            parallel max pooling operations with varying kernel sizes to 
            the same input feature map.

            This structure allows the model to gather local and global context 
            information at different receptive field sizes without changing 
            the input resolution. After pooling, all outputs are concatenated 
            along the channel axis with the original input to form a rich 
            multi-scale feature representation.

            This technique is widely used in object detection models (like YOLOv5) 
            to improve robustness to object scale and position variance.

        Args:
            size1 (tuple): Kernel size for the first MaxPool layer. Default is (5,5).
            size2 (tuple): Kernel size for the second MaxPool layer. Default is (9,9).
            size3 (tuple): Kernel size for the third MaxPool layer. Default is (13,13).

        Input:
            A 4D tensor of shape (batch_size, height, width, channels)

        Output:
            A 4D tensor with shape (batch_size, height, width, channels * 4)

        Example:
            >>> layer = SPP()
            >>> y = layer(x)  # x: input tensor
    """

    def __init__(self, size1=(5,5), size2=(9,9), size3=(13,13)):
        super().__init__()
        self.size1 = size1
        self.size2 = size2
        self.size3 = size3

    def build(self, input_shape):
        self.maxpool1 = MaxPool2D(self.size1, strides= 1, padding="same")
        self.maxpool2 = MaxPool2D(self.size2, strides= 1, padding="same")
        self.maxpool3 = MaxPool2D(self.size3, strides= 1, padding="same")

    def call(self, input):
        mp1 = self.maxpool1(input)
        mp2 = self.maxpool2(input)
        mp3 = self.maxpool3(input)

        x = tf.concat([input, mp1, mp2, mp3], axis= -1)
        return x