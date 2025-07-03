import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from architecture import neck  # import your neck class (which includes FPN + PANet)

# Step 1: Simulate backbone outputs (as Inputs)
c3 = Input(shape=(80, 80, 512), name="C3_Input")    # Shallow features
c4 = Input(shape=(40, 40, 1024), name="C4_Input")   # Mid-level features
c5 = Input(shape=(20, 20, 2048), name="C5_Input")   # Deep features

# Step 2: Instantiate neck and get 3 fused feature maps
neck_layer = neck()
p3, p4, p5 = neck_layer(c3, c4, c5)

# Step 3: Wrap neck into a Keras model for visualization & testing
model = Model(inputs=[c3, c4, c5], outputs=[p3, p4, p5], name="CustomNeckModel")
# Step 4: Print the model summary
model.summary(line_length=160)
