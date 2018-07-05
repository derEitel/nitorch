from sklearn.metrics import recall_score


def specificity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=0)


def sensitivity(y_true, y_pred):
    return recall_score(y_true, y_pred, pos_label=1)


def balanced_accuracy(y_true, y_pred):
    specificity = specificity(y_true, y_pred)
    sensitivity = sensitivity(y_true, y_pred)
    return (specificity + sensitivity) / 2
