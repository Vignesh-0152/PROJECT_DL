import tensorflow as tf
from tensorflow.keras import Model, Input
from architecture import CNN  # Replace with correct import path if needed

# Number of classes
cls = 3
# Create functional model wrapper around your CNN(Layer)
input_tensor = Input(shape=(640, 640, 3))
h3, h4, h5 = CNN(cls)(input_tensor)

# Create full Keras Model for summary and training
model = Model(inputs=input_tensor, outputs=[h3, h4, h5], name="CustomObjectDetector")

# Print model summary
model.summary(line_length= 200)

# Run a test forward pass with dummy input
dummy_input = tf.random.normal((1, 640, 640, 3))
outputs = model(dummy_input)

# Print output shapes
print("\nOutput Shapes:")
for i, out in enumerate(outputs, start=3):
    print(f"h{i}: {out.shape}")
