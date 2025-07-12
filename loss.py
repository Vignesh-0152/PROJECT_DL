from loss import bbloss, classloss, objloss
from tensorflow.keras.layers import Input
y_pred = y_true = Input(shape=(8400,8), batch_size= 8)
x = bbloss()(y_true, y_pred)
y = classloss()(y_true, y_pred)
z = objloss()(y_true, y_pred)

print(x.shape)