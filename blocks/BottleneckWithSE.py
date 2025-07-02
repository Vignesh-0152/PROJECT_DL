import tensorflow as tf
from tensorflow.keras.layers import Layer,Activation
from tensorflow.keras.regularizers import l2
from .Bottleneck import Bottleneck
from .SEBlock import SEBlock

class BottleneckWithSE(Layer):
    """
        Bottleneck with Squeeze-and-Excitation (SE) Block:
            This composite block integrates a standard Bottleneck structure with a 
            Squeeze-and-Excitation mechanism to enhance channel-wise feature selection. 
            The Bottleneck first reduces the channel dimensions using a 1x1 convolution, 
            processes spatial features with a 3x3 convolution, and then restores the
            original channel size via another 1x1 convolution. After this transformation, 
            an SE block is applied to adaptively recalibrate channel-wise responses 
            by modeling global context using global average pooling and fully connected 
            layers. Finally, a residual connection adds the original input to the 
            recalibrated output, followed by a Swish activation to introduce non-linearity. 
            This design improves representational power while maintaining efficiency.

        Args:
            filtersForBottleneck(int): Input filter size for the bottleneck layer.(default: None)
            filtersForSEBlock(int): Input filter size for SEBlock placed after the Bottleneck.(default: None)
            reductionRateBottleneck(int): Reduction rate for Bottleneck.( default: 2)
            reductionRateSEBlock(int): Reduction rate for SEBlock.( default: 4)
        
        Input: 
            Input from previous layer of shape(x,x,y)

        Output:
            output of shape(x,x,y)
    """

    
    def __init__(
                    self, filtersForBottleneck = None, 
                    filtersForSEBlock = None, 
                    reductionRateBottleneck = 2, 
                    reductionRateSEBlock = 4
                ):        
        super().__init__()
        self.filters1 = filtersForBottleneck
        self.filters2 = filtersForSEBlock
        self.reductionRate1 = reductionRateBottleneck
        self.reductionRate2 = reductionRateSEBlock

    def build(self, input_shape):
        self.bottleneck = Bottleneck(self.filters1, int(self.reductionRate1 // 2))
        self.se = SEBlock(self.filters2, self.reductionRate2)
        self.activation = Activation("swish")


    def call(self, input):
        x = self.bottleneck(input)
        x = self.se(x)
        x = tf.add(input, x)
        x = self.activation(x)

        return x