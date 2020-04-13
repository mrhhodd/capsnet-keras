
from tensorflow.keras import backend as K

def specificity(y_true, y_pred):
    [_, tn, fp, _] = _analyse_data(y_true, y_pred)
    return tn / (tn + fp + K.epsilon())


def sensitivity(y_true, y_pred):
    [tp, _, _, fn] = _analyse_data(y_true, y_pred)
    return tp / (tp + fn + K.epsilon())    


def f1_score(y_true, y_pred):
    [tp, tn, fp, fn] = _analyse_data(y_true, y_pred)
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * precision * recall / (precision + recall + K.epsilon())


def _analyse_data(y_true, y_pred):
    y_pred = K.argmax(y_pred)
    true_positive = y_true * y_pred
    true_negative = (1 - y_true) * (1 - y_pred)
    false_positive = (1 - y_true) * y_pred
    false_negative = y_true * (1 - y_pred)
    return true_positive, true_negative, false_positive, false_negative
