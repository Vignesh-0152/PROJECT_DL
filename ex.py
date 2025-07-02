import tensorflow as tf
from architecture.backbone import backbone

# Create dummy input
input_tensor = tf.keras.Input(shape=(640, 640, 3))

# Pass it through the backbone
backbone_layer = backbone()
c3, c4, c5 = backbone_layer(input_tensor)

# Inspect internal layers (layer-wise summary)
model = tf.keras.Model(inputs=input_tensor, outputs=[c3, c4, c5])
model.summary(line_length=160)
