import tensorflow as tf
from tensorflow.keras.layers import Layer

class flattenandconcatenate(Layer):
    def __init__(self):
        super().__init__()

    def call(self, x_80, x_40, x_20):
        # Remove print statements for graph compatibility
        x_80 = tf.reshape(x_80, (tf.shape(x_80)[0], -1, tf.shape(x_80)[-1]))
        x_40 = tf.reshape(x_40, (tf.shape(x_40)[0], -1, tf.shape(x_40)[-1]))
        x_20 = tf.reshape(x_20, (tf.shape(x_20)[0], -1, tf.shape(x_20)[-1]))

        # Ensure all shapes are compatible for concatenation
        return tf.concat([x_80, x_40, x_20], axis=1)