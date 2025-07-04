import tensorflow as tf
from tensorflow.keras import Input, Model
from architecture import head  # make sure import path is correct!

# Number of classes
cls = 3

# Define dummy PANet outputs for p3, p4, p5
# Assuming channel size = 256 from your neck output
p3_input = Input(shape=(80, 80, 256), name='p3_input')
p4_input = Input(shape=(40, 40, 256), name='p4_input')
p5_input = Input(shape=(20, 20, 256), name='p5_input')

# Create head layer instance
head_layer = head(cls)

# Get head outputs
h3, h4, h5 = head_layer(p3_input, p4_input, p5_input)

# Wrap in model
test_head_model = Model(inputs=[p3_input, p4_input, p5_input],
                        outputs=[h3, h4, h5],
                        name='TestHeadLayer')

# Print summary
test_head_model.summary(line_length= 200)

# Run dummy test pass
dummy_p3 = tf.random.normal((1, 80, 80, 256))
dummy_p4 = tf.random.normal((1, 40, 40, 256))
dummy_p5 = tf.random.normal((1, 20, 20, 256))

outputs = test_head_model([dummy_p3, dummy_p4, dummy_p5])

# Print output shapes
print("\nOutput Shapes:")
for i, out in enumerate(outputs, start=3):
    print(f"h{i}: {out.shape}")
