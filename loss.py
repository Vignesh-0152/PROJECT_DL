import tensorflow as tf
from loss import bbloss, classloss, objloss, CustomLoss

# Simulate fake ground truth and prediction
y_true = tf.random.uniform((1, 8400, 8))
y_pred = tf.random.uniform((1, 8400, 8))

# Call the custom loss functions
x = bbloss()(y_true, y_pred)
y = classloss()(y_true, y_pred)
z = objloss()(y_true, y_pred)
a = CustomLoss()(y_true, y_pred)
# Print actual scalar loss values
print("BBox Loss:", x.shape, type(y), y.dtype)
print("Class Loss:", y.shape, type(y), y.dtype)
print("Obj Loss:", z.shape, type(z), y.dtype)
print("custom Loss:", a.shape, type(a), a.dtype)
