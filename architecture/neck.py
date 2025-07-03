from blocks import FPN, PANet
from tensorflow.keras.layers import Layer

class neck(Layer):
    def __init__(self):
        super().__init__()
        self.fpn = FPN()
        self.pan = PANet()

    def call(self, c3, c4, c5):
        p5, p4, p3 = self.fpn(c5, c4, c3)
        p3, p4, p5 = self.pan(p3, p4, p5)

        return p3, p4, p5