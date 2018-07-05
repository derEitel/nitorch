from sklearn.metrics import recall_score


def balanced_accuracy(y_true, y_pred):
    specificity = specificity(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return (specificity + recall) / 2


def specificity(y_true, y_pred):
	return recall_score(y_true, y_pred, pos_label=0)


def recall(y_true, y_pred):
	return recall_score(y_true, y_pred, pos_label=1)
