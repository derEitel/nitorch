import torch
from torch import nn


def predict(
        all_outputs,
        prediction_type,
        criterion,
        **kwargs
):
    """ Predict according to loss and prediction type."""
    if prediction_type == "binary":
        all_preds = classif_inference(all_outputs, criterion=criterion, **kwargs)

    elif prediction_type == "classification":
        # TODO: develop inference
        raise NotImplementedError("Multiclass-classification \
            not yet implemented")
    elif prediction_type in ["regression", "reconstruction", "variational"]:
        # TODO: test different loss functions
        all_preds = all_outputs.data
    else:
        raise NotImplementedError

    return all_preds


def classif_inference(
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
