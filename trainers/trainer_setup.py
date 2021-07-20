from tensorflow.keras.optimizers import SGD, Adamax, Adam, Adadelta, Adagrad, Nadam
from tensorflow.keras.regularizers import l1, l2
from utils.losses import lp_loss


def lr_exp_decay(epoch, lr):
    k = 0.01
    return initial_learning_rate * np.exp(-k * epoch)

optimizers = {
    'adam': Adam,
    'sgd': SGD,
    'adamax': Adamax,
    'adadelta': Adadelta,
    'adagrad': Adagrad,
    'nadam': Nadam
}

regularizers = {
    'l1': l1,
    'l2': l2
}

losses = {
    "lploss": lp_loss
}