import tensorflow as tf
from tensorflow.keras import backend as K

EPSILON = K.epsilon()

def specificity(y_true, y_pred):
    [_, tn, fp, _] = _analyse_data(y_true, y_pred)
    return K.mean(tn / (tn + fp + EPSILON))


def sensitivity(y_true, y_pred):
    [tp, _, _, fn] = _analyse_data(y_true, y_pred)
    return K.mean(tp / (tp + fn + EPSILON))   


def f1_score(y_true, y_pred):
    [tp, tn, fp, fn] = _analyse_data(y_true, y_pred)
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    return K.mean(2 * precision * recall / (precision + recall + EPSILON))


def _analyse_data(y_true, y_pred):
    n_class = y_pred.shape[1]
    y_pred = K.one_hot(K.argmax(y_pred, axis=1), n_class)
    true_positive = K.sum(y_true * y_pred, axis=1)
    true_negative = K.sum((1 - y_true) * (1 - y_pred), axis=1)
    false_positive = K.sum((1 - y_true) * y_pred, axis=1)
    false_negative = K.sum(y_true * (1 - y_pred), axis=1)
    return true_positive, true_negative, false_positive, false_negative
