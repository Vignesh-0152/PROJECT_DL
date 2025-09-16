import tensorflow as tf

class xyxycal:
    def __init__(self):
        pass
    
    def __call__(self, ypred_ytrue, cls):
        return self.call(ypred_ytrue= ypred_ytrue, cls= cls)

    def call(self, ypred_ytrue, cls):
        boxes = ypred_ytrue[... , :4]
        ob_score = ypred_ytrue[..., 4]
        cls_score = ypred_ytrue[..., 5:]

        cx = boxes[... , 0]
        cy = boxes[... , 1]
        w = boxes[... , 2]
        h = boxes[... , 3]

        x1 = tf.expand_dims(cx - w / 2.0, axis= -1)
        x2 = tf.expand_dims(cx + w / 2.0, axis= -1)
        y1 = tf.expand_dims(cy - h / 2.0, axis= -1)
        y2 = tf.expand_dims(cy + h / 2.0, axis= -1)

        boxes = tf.stack([y1, x1, y2, x2], axis= -1)
        boxes = tf.expand_dims(boxes, axis= 2)
        boxes = tf.tile(boxes, [1,1,cls,1])

        return boxes, ob_score, cls_score
