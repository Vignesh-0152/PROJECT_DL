import tensorflow as tf
from os import listdir
from os.path import join, exists
import numpy as np
from utils import flattenandconcatenate

class classprocess():
    """
        Class Number Extraction:
            This class takes input of Label-path for Training, Testing or Validation 
            and extract the maximum class number from the label files.

        Input:
            y_train: (path) The path for Label

        Output:
            cls: (int) The maximum class the input labels have
        Example:
        >>> dataset = classnumber(label_path)
        >>> print(type(dataset))
            <preprocess.classnumber.classnumber object at 0x00000242A4347C40>
    """
    def __init__(self, y_train):
        self.y_train = y_train
        self.cls = self.process_class()

    def process_class(self):
        max_cls = tf.cast(-1.0, dtype = tf.float32)

        for fname in listdir(self.y_train):
            if not fname.lower().endswith('.txt'):
                continue
            
            label_path = join(self.y_train, fname)

            if exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        try:
                            cls, cx, cy, w, h = map(float, line.strip().split())
                        except ValueError:
                            print("🚨 Bad label format in file:", label_path, "Line:", line)
                            continue
                        cls = float(cls)
                        if cls > max_cls:
                            max_cls = cls

        return tf.cast(max_cls + 1, tf.int32)  # Adding 1 to convert from max index to count
        