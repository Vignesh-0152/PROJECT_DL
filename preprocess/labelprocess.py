import tensorflow as tf
from os import listdir
from os.path import join, exists
import numpy as np
from utils import flattenandconcatenate

class labelprocess():
    """
        Label Pre-Processing:
            This class takes input of Label-path for Training, Testing or Validation 
            and convert them into python tuple. Each label need to converted into Tensor 
            to be used by training, testing and validation by tensorflow.keras.       

        Input:
            y_train: (path) The path for Label
    """

    def __init__(self, y_train, classnumber):
        self.y_train = y_train
        self.classnumber = classnumber
        print("This is a auto call class and Label pre_processing is started:😎✌️")  
        self.flatandconcat = flattenandconcatenate() 
        self.label = self.process_label()

    def process_label(self, input_size = 640):
        y_20, y_40, y_80 = [], [], []

        for fname in listdir(self.y_train):
            if not fname.lower().endswith('.txt'):
                continue

            label_file = join(self.y_train, fname)

            # Zero matrices with (5 + self.classnumber) channels
            label_80 = tf.zeros((80, 80, 1, 5 + self.classnumber), tf.float32)
            label_40 = tf.zeros((40, 40, 1, 5 + self.classnumber), tf.float32)
            label_20 = tf.zeros((20, 20, 1, 5 + self.classnumber), tf.float32)

            if exists(label_file):
                with open(label_file, "r") as f:
                    for line in f:
                        try:
                            cls, cx, cy, w, h = map(float, line.strip().split())
                            cls = int(cls)
                        except ValueError:
                            continue

                        # Convert class to one-hot
                        onehot_cls = tf.one_hot(cls, depth=self.classnumber, dtype=tf.float32)

                        # Convert relative coords → actual pixels
                        W, H = w * input_size, h * input_size
                        avg_dim = tf.reduce_mean([W, H])

                        # Select grid size
                        if avg_dim <= 80:
                            grid_size = 80    # label = label_80

                        elif avg_dim <= 200:
                            grid_size = 40    # label = label_40

                        elif avg_dim > 200:
                            grid_size = 20   

                        # Grid cell positions
                        cx_cell, cy_cell = cx * grid_size, cy * grid_size
                        gx, gy = tf.cast(tf.floor(cx_cell), tf.int32), tf.cast(tf.floor(cy_cell), tf.int32)
                        cx_grid, cy_grid = cx_cell - tf.cast(gx, tf.float32), cy_cell - tf.cast(gy, tf.float32)

                        # Output vector: [cx_grid, cy_grid, w, h, conf, onehot_classes...]
                        output = tf.concat([[cx_grid, cy_grid, w, h, 0.9], onehot_cls], axis=0)

                        index = tf.convert_to_tensor([[gy, gx, 0]], dtype=tf.int32)
                        update = tf.expand_dims(output, axis=0)  # make it batch shape

                        # Update correct grid
                        if grid_size == 80:
                            label_80 = tf.tensor_scatter_nd_update(label_80, index, update)
                        elif grid_size == 40:
                            label_40 = tf.tensor_scatter_nd_update(label_40, index, update)
                        elif grid_size == 20:
                            label_20 = tf.tensor_scatter_nd_update(label_20, index, update)

            y_20.append(label_20)
            y_40.append(label_40)
            y_80.append(label_80)

        # Stack outputs
        y_20 = tf.stack(y_20)
        y_40 = tf.stack(y_40)
        y_80 = tf.stack(y_80)
        y = self.flatandconcat(y_80, y_40, y_20)        
        print("Label pre_processing is done and values are returned:😎✌️")
        return y

