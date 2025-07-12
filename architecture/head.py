import tensorflow as tf
from tensorflow.keras.layers import Layer
from blocks import headblock

class head(Layer):
    
    def __init__(self, cls):
        super().__init__()
        self.cls = cls
        self.headblock_p3 = headblock(cls)
        self.headblock_p4 = headblock(cls)
        self.headblock_p5 = headblock(cls)

    def call(self, p3, p4, p5):

        p3_box, p3_class, p3_object = self.headblock_p3(p3)
        p4_box, p4_class, p4_object = self.headblock_p4(p4)
        p5_box, p5_class, p5_object = self.headblock_p5(p5)
        
        h3 = tf.concat([p3_box, p3_object, p3_class], axis = -1)
        h4 = tf.concat([p4_box, p4_object, p4_class], axis = -1)
        h5 = tf.concat([p5_box, p5_object, p5_class], axis = -1)

        return h3, h4, h5