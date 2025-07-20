import tensorflow as tf
from .xyxymap50 import xyxycal

class mAP50():
    def __init__(self, cls):
        self.no_of_cls = cls
        self.xyxy = xyxycal()

    def __call__(self, ytrue, ypred):
        return self.call(ytrue, ypred)

    def call(self, ytrue, ypred):

        boxes, ob_score, cls_score = self.xyxy(ypred, self.no_of_cls)

        if(ob_score.shape == cls_score.shape):
            score = ob_score * cls_score
        else:
            score = ob_score[..., tf.newaxis] * cls_score

        nms_output = tf.image.combined_non_max_suppression(
            boxes= boxes,
            scores= score,
            max_output_size_per_class= 100,
            max_total_size= self.cls * 100,
            iou_threshold= 0.5,
            score_threshold= 0.5,
            clip_boxes= False
        )

        box = nms_output.nmsed_boxes                                            #[B, N, 4]
        cls = nms_output.nmsed_classes                                          #[B, N]
        scores = tf.cast(nms_output.nmsed_scores, tf.int32)                     #[B, N]
        valid = nms_output.valid_detections                                     #[N]

        gt_box, gt_ob_score, gt_cls_score = self.xyxy(ypred_ytrue= ytrue, cls= self.no_of_cls)
        if(gt_ob_score.shape == gt_cls_score.shape):
            gt_score = gt_ob_score * gt_cls_score
        else:
            gt_score = gt_ob_score[..., tf.newaxis] * gt_cls_score

        for i in range(self.no_of_cls):
            #class matchning to seperate based on classes
            cls_indices = tf.where(cls == i)
            gt_cls_indices = tf.where(gt_cls_score == i)

            #y_pred matching
            #consider the box which only match that particular class
            y_pred_boxes = tf.gather(box, cls_indices)
            y_pred_scores = tf.gather(scores, cls_indices)


            #y_true matching
            #consider the box which only match that particular class
            y_true_boxes = tf.gather(gt_box, gt_cls_indices)
            y_true_scores = tf.gather(gt_ob_score, gt_cls_indices)

            y_pred_boxes = tf.squeeze(y_pred_boxes, axis= 1)
            y_pred_scores = tf.squeeze(y_pred_scores, axis= 1)
            y_true_boxes = tf.squeeze(y_true_boxes, axis= 1)
            y_true_scores = tf.squeeze(y_true_scores, axis= 1)

            # Sort predictions by score descending
            sorted_indices = tf.argsort(y_pred_scores, direction='DESCENDING')
            y_pred_boxes = tf.gather(y_pred_boxes, sorted_indices)

            matched_gt = tf.zeros(tf.shape(y_true_boxes)[0], dtype=tf.bool)

            tp = []
            fp = []

            for pred_box in y_pred_boxes:
                ious = self._compute_iou(pred_box, y_true_boxes)  # shape [num_gt]
                max_iou = tf.reduce_max(ious)
                max_idx = tf.argmax(ious)

                if max_iou >= 0.5 and not matched_gt[max_idx]:
                    tp.append(1.0)
                    fp.append(0.0)
                    matched_gt = tf.tensor_scatter_nd_update(matched_gt, [[max_idx]], [True])
                else:
                    tp.append(0.0)
                    fp.append(1.0)

            tp = tf.cumsum(tp)
            fp = tf.cumsum(fp)
            total_gt = tf.cast(tf.shape(y_true_boxes)[0], tf.float32)

            recall = tp / (total_gt + 1e-6)
            precision = tp / (tp + fp + 1e-6)

            # Compute AP using trapezoidal rule (simple method)
            ap = self._compute_ap(precision, recall)

            if i == 0:
                all_ap = tf.expand_dims(ap, axis=0)
            else:
                all_ap = tf.concat([all_ap, tf.expand_dims(ap, axis=0)], axis=0)

        return tf.reduce_mean(all_ap)

    def _compute_iou(self, box, boxes):
        """
        Compute IoU between a single box and multiple boxes.
        box: [4], boxes: [N, 4]
        """
        x1 = tf.maximum(box[0], boxes[:, 0])
        y1 = tf.maximum(box[1], boxes[:, 1])
        x2 = tf.minimum(box[2], boxes[:, 2])
        y2 = tf.minimum(box[3], boxes[:, 3])

        inter_area = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

        union_area = box_area + boxes_area - inter_area
        return inter_area / (union_area + 1e-6)

    def _compute_ap(self, precision, recall):
        """
        Compute Average Precision (AP) from precision and recall arrays.
        """
        precision = tf.concat([[0.0], precision, [0.0]], axis=0)
        recall = tf.concat([[0.0], recall, [1.0]], axis=0)

        # Make the precision curve non-increasing
        for i in range(tf.shape(precision)[0] - 2, -1, -1):
            precision = tf.tensor_scatter_nd_update(
                precision, [[i]], [tf.maximum(precision[i], precision[i + 1])]
            )

        indices = tf.where(recall[1:] != recall[:-1])[:, 0]
        ap = tf.reduce_sum(
            (tf.gather(recall, indices + 1) - tf.gather(recall, indices)) *
            tf.gather(precision, indices + 1)
        )

        return ap
