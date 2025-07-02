from tensorflow.keras.layers import Layer
from blocks import conv2d, ResNetWithBottleneck, c2f, SPP, SEBlock

class backbone(Layer):
    """
        Custom CNN Backbone for Object Detection:
            This backbone combines several powerful blocks such as ResNet-inspired 
            residual units, C2F fusion, SPP for multi-scale spatial context, and 
            SE (Squeeze-and-Excitation) attention modules. It processes the input 
            image and outputs three feature maps at different resolutions (C3, C4, C5),
            ideal for downstream detection heads like FPN, PANet, or YOLO heads.

        Architecture:
            Input -> Conv(32, 1x1, s=2)  
                -> Conv(64, 1x1, s=1)  
                -> ResNetWithBottleneck  
                -> Conv(128, 1x1, s=2)  
                -> C2F (split=4)  
                -> Conv(512, 3x3) → C3  
                -> Conv(256, 1x1, s=2)  
                -> Conv(256, 3x3)  
                -> SPP  
                -> SEBlock(1024) → C4  
                -> Conv(2048, 1x1, s=2)  
                -> Conv(2048, 3x3)  
                -> SEBlock(2048) → C5

        Input:
            A 4D Tensor of shape (batch_size, height, width, 3)

        Output:
            Tuple of 3 feature maps:
                - C3: (batch_size, 80, 80, 512)
                - C4: (batch_size, 40, 40, 1024)
                - C5: (batch_size, 20, 20, 2048)
    """

    def __init__(self):
        """
            Initializes the custom backbone layers using modular components from `blocks` package.
        """
        super().__init__()

    def build(self, input_shape):
        """
            Build all the layers with dynamic input shape.

            Args:
                input_shape (tf.TensorShape): Shape of the input tensor
        """
        self.conv1 = conv2d(filters = 32, kernel_size=(1,1), strides = 2)
        self.conv2 = conv2d(filters = 64, kernel_size=(1,1), strides = 1)
        self.resnetwithbottlenck = ResNetWithBottleneck(filters = 64)
        self.conv3 = conv2d(filters = 128, kernel_size = (1,1), strides = 2)
        self.c2f = c2f(filters = 256, splitNumber = 4)
        self.conv4 = conv2d(filters = 512, kernel_size = (3,3), strides = 1)    # c3
        self.conv5 = conv2d(filters = 256, kernel_size = (1,1), strides = 2)
        self.conv6 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.spp = SPP()
        self.se1 = SEBlock(filters = 1024, reductionRate = 4)                   #c4
        self.conv7 = conv2d(filters = 2048, kernel_size = (1,1), strides = 2)
        self.conv8 = conv2d(filters = 2048, kernel_size =(3,3), strides = 1)
        self.se2 = SEBlock(filters = 2048, reductionRate = 4)                   #c5

    def call(self, input):
        """
            Forward pass of the backbone model.

            Args:
                input (tf.Tensor): Input image tensor of shape (batch_size, height, width, 3)

            Returns:
                Tuple[tf.Tensor, tf.Tensor, tf.Tensor]: Feature maps C3, C4, C5
        """
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.resnetwithbottlenck(x)
        x = self.conv3(x)
        x = self.c2f(x)
        c3 = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.spp(x)
        x = c4 = self.se1(x)
        x = self.conv7(x)
        x = self.conv8(x)
        c5 = self.se2(x)

        return c3, c4, c5
