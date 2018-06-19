from sklearn.metrics import precision_score, recall_score

def balanced_accuracy(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return (precision + recall) / 2