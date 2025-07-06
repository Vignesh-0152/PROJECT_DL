import tensorflow as tf

class xycalc:
    def __init__(self, cx, cy, w, h):
        self.cx = cx
        self.cy = cy
        self.w = w
        self.h = h

        self.x1, self.y1, self.x2, self.y2 = self.call()

    def call(self):
        x1 = self.cx - self.w / 2.0
        x2 = self.cx + self.w / 2.0
        y1 = self.cy - self.h / 2.0
        y2 = self.cy + self.h / 2.0

        x1 = tf.clip_by_value(x1, 0.0, 640.0)
        y1 = tf.clip_by_value(y1, 0.0, 640.0)
        x2 = tf.clip_by_value(x2, 0.0, 640.0)
        y2 = tf.clip_by_value(y2, 0.0, 640.0)

        return x1, y1, x2, y2