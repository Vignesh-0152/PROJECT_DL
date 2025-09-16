import tensorflow as tf
from os import listdir
from os.path import join, exists
import numpy as np
from utils import flattenandconcatenate

class imgprocess():
    """
        Image Pre-Processing:
            This class takes input of Image-path and Label-path for Training, Testing or Validation 
            and convert them into python tuple. Each images and labels need to converted into Tensor 
            to be used by training, testing and validation by tensorflow.keras.

        Input:
            x_train: (path) The path for Image
            y_train: (path) The path for Label
            input_size: (int) The input image size(optional and default value is set to 640)

        Output:
            image: (Tensor) An Tensor contains Normalized values[0-1] of images
            label: (tuple) contains 3 Tensors each represent the label for 3 different size of boxes
            cls: (int) The maximum class the input labels have

        Example:
        >>> dataset = imgprocess(img_path, label_path, 640)
        >>> print(type(dataset))
            <preprocess.imgprocess.imgprocess object at 0x00000242A4347C40>
    """

    def __init__(self, x_train, input_size = 640):
        self.x_train = x_train
        self.input_size = input_size
        print("This is a auto call class and Image pre_processing is started:😎✌️")
        self.image = self.process_images()        

    def process_images(self):
        x = []
        for fname in listdir(self.x_train):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            img_path = join(self.x_train, fname)
            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels=3)
            img = tf.image.resize(img, (self.input_size, self.input_size))
            img = img / 255.0
            img = tf.cast(img, tf.float32)

            x.append(img)

        x = tf.stack(x)
        print("Label pre_processing is done and values are returned:😎✌️")
        print("Shape of images:", x.shape)
        return x
