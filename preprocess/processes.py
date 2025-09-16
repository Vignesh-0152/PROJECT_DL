from .imgprocess import imgprocess
from .labelprocess import labelprocess

class process:
    def __init__(self, x_train, y_train, classnumber):
        self.x_train = x_train
        self.y_train = y_train
        self.classnumber = classnumber

        self.img_process = imgprocess(x_train= x_train)
        self.label_process = labelprocess(y_train= y_train, classnumber= classnumber)

        self.image, self.label = self.preprocessing()

    def preprocessing(self):
        return self.img_process.image, self.label_process.label
