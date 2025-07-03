import tensorflow as tf
from tensorflow.keras.layers import Layer, Dropout
from .conv2d import conv2d
from .upsampleandadd import upsampleandadd

class FPN(Layer):
    """
        Feature Pyramid Network (FPN):
            This module fuses multi-scale feature maps from the backbone (C3, C4, C5) 
            into a unified set of high-level semantic feature maps (P3, P4, P5) while 
            maintaining spatial resolution. It enables robust object detection across 
            scales (small to large objects).

        Logic:
            Step 1: Apply 1x1 Conv2D to compress channels of C5, C4, and C3 to a fixed size (256)
            Step 2: 
                p5 = Conv1x1(C5)
                p4 = UpsampleAndAdd(Conv1x1(C4), p5)
                p3 = UpsampleAndAdd(Conv1x1(C3), p4)
            Step 3: Apply 3x3 Conv2D to p3, p4, and p5 for feature smoothing
            Step 4: Apply Dropout(0.05) for regularization

        Args:
            None

        Input:
            c5: tensor from backbone (typically deepest feature map), shape = (B, H/32, W/32, C)
            c4: intermediate feature map, shape = (B, H/16, W/16, C)
            c3: shallowest high-level feature map, shape = (B, H/8, W/8, C)

        Output:
            p3: Fused high-resolution feature map, shape = (B, H/8, W/8, 256)
            p4: Fused mid-resolution feature map, shape = (B, H/16, W/16, 256)
            p5: Fused low-resolution feature map, shape = (B, H/32, W/32, 256)

        Notes:
            - The 1x1 convs ensure channel alignment across all inputs.
            - The upsample-and-add structure supports top-down fusion from deep to shallow layers.
            - Outputs are ordered from fine (p3) to coarse (p5) for compatibility with PANet and detection heads.
            - Designed to be compatible with YOLOv8-style anchor-free models.

    """

    def __init__(self):
        super().__init__()
    
    def build(self):
        self.convc5 = conv2d(filters = 256, kernel_size = (1,1), strides = 1)
        self.upsample4 = upsampleandadd()
        self.upsample3 = upsampleandadd()
        self.convp5 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.convp4 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.convp3 = conv2d(filters = 256, kernel_size = (3,3), strides = 1)
        self.drop5 = Dropout(0.05)
        self.drop4 = Dropout(0.05)
        self.drop3 = Dropout(0.05)

    def call(self, c5, c4, c3):
        p5 = self.convc5(c5)
        p4 = self.upsample4(c4, p5)
        p3 = self.upsample3(c3, p4)

        p5 = self.convp5(p5)
        p4 = self.convp4(p4)
        p3 = self.convp3(p3)

        p5 = self.drop5(p5)
        p4 = self.drop4(p4)
        p3 = self.drop3(p3)

        return p5, p4, p3