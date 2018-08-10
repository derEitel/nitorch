import numpy
import torch
from torch import nn

def predict(
    outputs,
    labels,
    all_preds,
    all_labels,
    prediction_type,
    criterion,
    **kwargs
    ):
    """ Predict according to loss and prediction type."""
    if prediction_type == "binary":
        if isinstance(criterion, nn.BCEWithLogitsLoss):
            all_preds, all_labels = bce_with_logits_inference(
                outputs,
                labels,
                all_preds,
                all_labels,
                **kwargs
            )
        elif isinstance(criterion, nn.BCELoss):
            all_preds, all_labels = bce_inference(
                outputs,
                labels,
                all_preds,
                all_labels,
                **kwargs
            )
        return all_preds, all_labels
    elif prediction_type == "classification":
        # TODO: develop inference
        raise NotImplementedError("Multiclass-classification \
            not yet implemented")
    elif prediction_type == "regression":
        # TODO: test different loss functions
        all_preds, all_labels = regression_inference(
                outputs,
                labels,
                all_preds,
                all_labels
        )
        return all_preds, all_labels
    elif prediction_type == "reconstruction":
        # TODO: test different loss functions
        all_preds, all_labels = regression_inference(
                outputs,
                labels,
                all_preds,
                all_labels
        )
        return all_preds, all_labels
    elif prediction_type == "variational":
        # TODO: test different loss functions
        all_preds, all_labels = variational_inference(
                outputs,
                labels,
                all_preds,
                all_labels
        )
        return all_preds, all_labels
    else:
        raise NotImplementedError


def bce_with_logits_inference(outputs, labels, all_preds, all_labels, **kwargs):
    sigmoid = torch.sigmoid(outputs)
    if kwargs["class_threshold"]:
        class_threshold = kwargs["class_threshold"]
    else:
        class_threshold = 0.5
    print
    predicted = sigmoid.data >= class_threshold
    for j in range(len(predicted)):
        all_preds.append(predicted[j].cpu().numpy()[0])
        all_labels.append(labels[j].cpu().numpy()[0])
    return all_preds, all_labels

def bce_inference(outputs, labels, all_preds, all_labels, **kwargs):
    if kwargs["class_threshold"]:
        class_threshold = kwargs["class_threshold"]
    else:
        class_threshold = 0.5
    predicted = outputs.data >= class_threshold
    for j in range(len(predicted)):
        all_preds.append(predicted[j].cpu().numpy()[0])
        all_labels.append(labels[j].cpu().numpy()[0])
    return all_preds, all_labels

def regression_inference(outputs, labels, all_preds, all_labels):
    predicted = outputs.data
    # TODO: replace for loop with something faster
    for j in range(len(predicted)):
        all_preds.append(predicted[j].cpu().numpy()[0])
        all_labels.append(labels[j].cpu().numpy()[0])
    return all_preds, all_labels

def variational_inference(outputs, labels, all_preds, all_labels):
    """ Inference for variational autoencoders. """
    # VAE outputs reconstruction, mu and std
    predicted = outputs[0].data # select reconstruction only
    # TODO: replace for loop with something faster
    for j in range(len(predicted)):
        all_preds.append(predicted[j].cpu().numpy()[0])
        all_labels.append(labels[j].cpu().numpy()[0])
    return all_preds, all_labels