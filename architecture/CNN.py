from tensorflow.keras.layers import Layer
from .backbone import backbone
from .neck import neck
from .head import head

class CNN(Layer):
    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.backbone = backbone()
        self.neck = neck()
        self.head = head(cls)

    def call(self, input):
        c3, c4, c5 = self.backbone(input)
        p3, p4, p5 = self.neck(c3, c4, c5)
        h3, h4, h5 = self.head(p3, p4, p5)

        return h3, h4, h5