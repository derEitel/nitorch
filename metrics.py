from sklearn.metrics import recall_score

def balanced_accuracy(y_true, y_pred):
    specificity = recall_score(y_true, y_pred, pos_label=0)
    sensitivity = recall_score(y_true, y_pred, pos_label=1)

    return (specificity + sensitivity) / 2
