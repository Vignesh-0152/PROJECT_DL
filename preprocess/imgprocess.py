import tensorflow as tf
from os import listdir
from os.path import join, exists

class imgprocess():

    def __init__(self, x_train, y_train, input_size = 640):
        self.x_train = x_train
        self.y_train = y_train
        self.input_size = input_size
        print("This is a auto call class and Image pre_processing is started:😎✌️")
        self.image, self.label, self.cls = self.decoding_input()
    
    def decoding_input(self):
        x, y_20, y_40, y_80 = [], [], [], []
        max_cls = -1

        for fname in listdir(self.x_train):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue
            
            img_path = join(self.x_train, fname)
            label_path = join(self.y_train, fname.split('.')[0] + '.txt')

            img = tf.io.read_file(img_path)
            img = tf.image.decode_image(img, channels= 3)
            img = tf.image.resize(img, (self.input_size, self.input_size))
            img = (img / 255.0)
            img = tf.cast(img, tf.float32)

            label_80 = tf.zeros((80,80,1,6), tf.float32)
            label_40 = tf.zeros((40,40,1,6), tf.float32)
            label_20 = tf.zeros((20,20,1,6), tf.float32)

            if exists(label_path):
                with open(label_path, "r") as f:
                    for line in f:
                        cls, cx, cy, w, h = map(float, line.strip().split())
                        
                        if cls > max_cls:
                            max_cls = cls

                        W = w * self.input_size
                        H = h * self.input_size
                        avg_dim = tf.reduce_mean([W,H])

                        if avg_dim <= 80:
                            grid_size = 80    # label = label_80

                        elif avg_dim <= 200:
                            grid_size = 40    # label = label_40

                        elif avg_dim > 200:
                            grid_size = 20    # label = label_20

                        cx_cell = cx * grid_size
                        cy_cell = cy * grid_size
                        gx = tf.cast(tf.floor(cx_cell), tf.int32)
                        gy = tf.cast(tf.floor(cy_cell), tf.int32)
                        cx_grid = cx_cell - tf.cast(gx, tf.float32)
                        cy_grid = cy_cell - tf.cast(gy, tf.float32)

                        output = [cx_grid, cy_grid, w, h, 0.9, cls]

                        if grid_size == 80:
                            label_80 = tf.tensor_scatter_nd_update(
                                tensor= label_80,
                                indices= [[gy, gx, 0]],
                                updates= [output]
                            )

                        if grid_size == 40:
                            label_40 = tf.tensor_scatter_nd_update(
                                tensor= label_40,
                                indices= [[gy, gx, 0]],
                                updates= [output]
                            )

                        if grid_size == 20:
                            label_20 = tf.tensor_scatter_nd_update(
                                tensor= label_20,
                                indices= [[gy, gx, 0]],
                                updates= [output]
                            )

            x.append(img)
            y_20.append(label_20)
            y_40.append(label_40)
            y_80.append(label_80)

        x = tf.stack(x)
        y_20 = tf.stack(y_20)
        y_40 = tf.stack(y_40)
        y_80 = tf.stack(y_80)
        y = [y_20, y_40, y_80]

        print("Image pre_processing is done and values are returned:😎✌️")
        return x, y, max_cls
