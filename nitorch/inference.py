import torch
from torch import nn


def predict(
        all_outputs,
        all_labels,
        prediction_type,
        criterion,
        **kwargs
):
    """ Predict according to loss and prediction type."""
    if prediction_type == "binary":
        all_preds = binary_classif_inference(all_outputs, criterion=criterion, **kwargs)

    elif prediction_type == "classification":
        all_preds, all_labels = multi_classif_inference(all_outputs, all_labels, criterion=criterion, **kwargs)
        
    elif prediction_type in ["regression", "reconstruction", "variational"]:
        # TODO: test different loss functions
        all_preds = all_outputs.data
    else:
        raise NotImplementedError

    return all_preds, all_labels


def binary_classif_inference(
        all_outputs,
        criterion,
        **kwargs
):
    if isinstance(criterion, nn.BCEWithLogitsLoss):
        all_outputs = torch.sigmoid(all_outputs)

    if kwargs["class_threshold"]:
        class_threshold = kwargs["class_threshold"]
    else:
        class_threshold = 0.5
    all_preds = (all_outputs.data >= class_threshold)
    
    return all_preds


def multi_classif_inference(
        all_outputs,
        all_labels,
        criterion,
        **kwargs
):
    all_preds = torch.argmax(all_outputs.data, 1)    
    # convert the labels from one-hot vectors to class variables for metric calculations
    if isinstance(criterion, nn.BCELoss):
        all_labels = torch.argmax(all_labels.data, 1)
    # TODO : Test other loss types like NLL and BCEWithLogitsLoss
    return all_preds, all_labels