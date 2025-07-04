from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation
from .SEBlock import SEBlock
from .conv2d import conv2d

class headblock(Layer):

    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.seblock = SEBlock(filters = 256, reductionRate = 4)
        self.conv1 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.conv_box = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.conv_class = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.conv_object = conv2d(filters = 256, kernel_size = (3,3), strides = 1)

        self.conv_box_output = Conv2D(4, (1,1), strides = 1, padding = "same")

        self.conv_class_output = Conv2D(self.cls, (1,1), strides = 1, padding = "same")
        self.batchnorm_class_output = BatchNormalization()
        self.activation_class_output = Activation("sigmoid")

        self.conv_object_output = Conv2D(1, (1,1), strides = 1, padding = "same")
        self.batchnorm_object_output = BatchNormalization()
        self.activation_object_output = Activation("sigmoid")

    def call(self, input):
        px = self.seblock(input)
        px = self.conv1(px)

        #box detection:
        px_box = self.conv_box(px)
        px_box_output = self.conv_box_output(px_box)

        #class detection:
        px_class = self.conv_class(px)
        px_class_output = self.conv_class_output(px_class)
        px_class_output = self.batchnorm_class_output(px_class_output)
        px_class_output = self.activation_class_output(px_class_output)

        #objectness score:
        px_object = self.conv_object(px)
        px_object_output = self.conv_object_output(px_object)
        px_object_output = self.batchnorm_object_output(px_object_output)
        px_object_output = self.activation_object_output(px_object_output)

        return px_box_output, px_class_output, px_object_output

