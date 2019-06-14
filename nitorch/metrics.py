import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import recall_score, roc_curve, auc


def prepare_values(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        if y_true.is_cuda:
            y_true = y_true.to(torch.device("cpu"))
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        if y_pred.is_cuda:
            y_pred = y_pred.to(torch.device("cpu"))
        y_pred = y_pred.numpy()
    return y_true, y_pred


def specificity(y_true, y_pred):
    y_true, y_pred = prepare_values(y_true, y_pred)
    return recall_score(y_true, y_pred, pos_label=0)


def sensitivity(y_true, y_pred):
    y_true, y_pred = prepare_values(y_true, y_pred)
    return recall_score(y_true, y_pred, pos_label=1)


def binary_balanced_accuracy(y_true, y_pred):
    y_true, y_pred = prepare_values(y_true, y_pred)
    spec = specificity(y_true, y_pred)
    sens = sensitivity(y_true, y_pred)
    return (spec + sens) / 2


def auc_score(y_true, y_pred):
    y_true, y_pred = prepare_values(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    return auc(fpr, tpr)


def classif_accuracy(labels, preds):
    if isinstance(labels, list):  # multihead accuracy
        return [classif_accuracy(each_labels, each_preds) for each_labels, each_preds in zip(labels, preds)]
    else:
        correct = (labels.int() == preds.int()).sum()
        return (correct.float()/len(labels)).item()


def regression_accuracy(labels, preds):
    if isinstance(labels, list):  # multihead accuracy
        return [regression_accuracy(each_labels, each_preds) for each_labels, each_preds in zip(labels, preds)]
    else:
        return (1.-F.l1_loss(preds, target=labels)).item()
