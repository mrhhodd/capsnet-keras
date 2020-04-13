EPSILON = 1e-9

def specificity(y_true, y_pred):
    [_, tn, fp, _] = _analyse_data(y_true, y_pred)
    return tn / (tn + fp + EPSILON)


def sensitivity(y_true, y_pred):
    [tp, _, _, fn] = _analyse_data(y_true, y_pred)
    return tp / (tp + fn + EPSILON)    


def f1_score(y_true, y_pred):
    [tp, tn, fp, fn] = _analyse_data(y_true, y_pred)
    precision = tp / (tp + fp + EPSILON)
    recall = tp / (tp + fn + EPSILON)
    return 2 * precision * recall / (precision + recall + EPSILON)


def _analyse_data(y_true, y_pred):
    true_positive = y_true * y_pred
    true_negative = (1 - y_true) * (1 - y_pred)
    false_positive = (1 - y_true) * y_pred
    false_negative = y_true * (1 - y_pred)
    return true_positive, true_negative, false_positive, false_negative
