from tensorflow.keras.metrics import TruePositives, TrueNegatives, FalsePositives, FalseNegatives

EPSILON = K.epsilon()
threshold = 0.5
true_positive = TruePositives(threshold)
true_negative = TrueNegatives(threshold)
false_positive = FalsePositives(threshold)
false_negative = FalseNegatives(threshold)

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
    return K.mean(2.0 * precision * recall / (precision + recall + EPSILON))


def _analyse_data(y_true, y_pred):
    return true_positive.update_state(y_true, y_pred).result().numpy() \
      true_negative.update_state(y_true, y_pred).result().numpy() \
      false_positive.update_state(y_true, y_pred).result().numpy() \
      false_negative.update_state(y_true, y_pred).result().numpy() \
