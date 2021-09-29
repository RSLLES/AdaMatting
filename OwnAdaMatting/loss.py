import tensorflow as tf
from keras.losses import Loss, BinaryCrossentropy, MeanAbsoluteError


class AdaptiveTrimapLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="adaptiveTrimapLoss"):
        self.bce = BinaryCrossentropy(from_logits=False)
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None, eps=1e-6):
        gt_trimap = tf.slice(y_true, [0,0,0,0], [-1, -1, -1, 3])
        trimap, _, _ = y_pred
        return self.bce(y_true=gt_trimap, y_pred=trimap)

class AlphaLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="alphaLoss"):
        self.mae = MeanAbsoluteError()
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, sample_weight=None, eps=1e-12):
        gt_alpha =  tf.slice(y_true, [0,0,0,3], [-1, -1, -1, 1])
        # gt_grey = tf.slice(y_true, [0,0,0,1], [-1, -1, -1, 1])
        _, alpha, _ = y_pred

        # return tf.reduce_sum(tf.abs(alpha - gt_alpha)*gt_grey)/(tf.reduce_sum(gt_grey) + eps)
        return self.mae(gt_alpha, alpha)

class MultiTaskLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="multitaskloss"):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, y_true, y_pred, loss_trimap, loss_alpha, sample_weight=None, eps=1e-6):
        _, _, weights = y_pred

        log_s1_sqr = tf.squeeze(tf.slice(weights, [0,0], [1,1]))
        log_s2_sqr = tf.squeeze(tf.slice(weights, [1,0], [1,1]))
        s1_sqr = tf.exp(log_s1_sqr)
        s2 = tf.exp(0.5*log_s2_sqr)

        # Fusion
        return loss_trimap/s1_sqr + 2*loss_alpha/s2 + log_s1_sqr + log_s2_sqr

class AlternateLoss(Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="multitaskloss"):
        super().__init__(reduction=reduction, name=name)

    def __call__(self, loss_trimap, loss_alpha, switch, eps=1e-7):
        w1 = 1.0 if switch else eps
        w2 = eps if switch else 1.0
        return w1*loss_trimap + w2*loss_alpha
        