import tensorflow as tf
from tensorflow.keras import backend


def lp_loss(x, y, d=2, p=2, size_average=True, reduction=True):
    num = tf.shape(x)[0]
    x = tf.reshape(x, (num, -1))
    y = tf.reshape(y, (num, -1))
    diff_norms = tf.norm(x - y, axis=1, ord=p)
    y_norms = tf.norm(y, axis=1, ord=p)

    if reduction:
        if size_average:
            return backend.mean(diff_norms / y_norms)
        else:
            return backend.sum(diff_norms / y_norms)

    return diff_norms / y_norms