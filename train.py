from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow as tf
from preprocess import imgprocess
from architecture import CNN
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from loss import CustomLoss
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import Precision, Recall, AUC

import os 
import datetime

train_image = r"C:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\Vehicles_Detection.v8i.yolov8\train\images"
train_label = r"C:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\Vehicles_Detection.v8i.yolov8\train\labels"

validation_image = r"C:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\Vehicles_Detection.v8i.yolov8\train\images"
validation_label = r"C:\Users\svign_ggx9gjx\Desktop\soft_computing\VIGNESH\VIGNESH\yolov8-multiple-vehicle-detection-main\yolov8-multiple-vehicle-detection\Vehicles_Detection.v8i.yolov8\train\labels"

train = imgprocess(x_train= train_image, y_train= train_label)
validate = imgprocess(x_train= validation_image, y_train= validation_label)

no_of_class = train.cls
print(no_of_class)

input = Input(shape=(640,640,3))
output = CNN(no_of_class)(input)

model = Model(input, output)

learning_rate = PolynomialDecay(
    initial_learning_rate= 1e-3,
    decay_steps= 20000,
    end_learning_rate= 1e-6,
    power=2,
    name= "learning_rate"
)

optimizer = Adam(
    learning_rate= learning_rate
)

loss = CustomLoss()
model.compile(
    optimizer= optimizer,
    loss = loss,
    metrics= ['accuracy', Precision(), Recall(), AUC()],
)

model.fit(
    x= train.image,
    y= train.label,
    batch_size= 4,
    epochs= 200,
    validation_data= (validate.image, validate.label),
    validation_batch_size= 4
)

