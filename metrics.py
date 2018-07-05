from sklearn.metrics import recall_score


def balanced_accuracy(y_true, y_pred):
    specificity = recall_score(y_true, y_pred, pos_label=0)
    recall = recall_score(y_true, y_pred, pos_label=1)
    return (specificity + recall) / 2


def specificity(y_true, y_pred):
	return recall_score(y_true, y_pred, pos_label=0)


def recall(y_true, y_pred):
	return recall_score(y_true, y_pred, pos_label=1)
