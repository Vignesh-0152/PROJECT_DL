from tensorflow.keras.models import Model
from tensorflow.keras import Input
import tensorflow as tf
from preprocess import classprocess, process
from architecture import CNN
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.optimizers import Adam
from loss import CustomLoss
from callbacks import CustomCallback, LiveTerminalOutput
from tensorflow.keras.metrics import Precision, Recall, AUC

import os 
import datetime

train_image = r"/kaggle/input/my-yolo-dataset/valid/images"
train_label = r"/kaggle/input/my-yolo-dataset/valid/labels"

validation_image = r"/kaggle/input/my-yolo-dataset/test/images"
validation_label = r"/kaggle/input/my-yolo-dataset/test/labels"

no_of_class = classprocess(train_label)

train = process(train_image, train_label, no_of_class.clss)
validate = process(validation_image, validation_label, no_of_class.clss)

input = Input(shape=(640,640,3))
output = CNN(no_of_class.clss)(input)

model = Model(input, output)

learning_rate = PolynomialDecay(
    initial_learning_rate= 1e-2,
    decay_steps= 60000,
    end_learning_rate= 1e-5,
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

callbacks = CustomCallback().Callback_list
callbacks.append(LiveTerminalOutput())

model.fit(
    x= train.image,
    y= train.label,
    batch_size= 8,
    epochs= 200,
    validation_data= (validate.image, validate.label),
    validation_batch_size= 4,
    callbacks= callbacks
)

