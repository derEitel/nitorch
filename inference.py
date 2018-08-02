import numpy
import torch

def predict(
    outputs,
    labels,
    all_preds,
    all_labels,
    prediction_type,
    criterion
    ):
    """ Predict according to loss and prediction type."""
    if prediction_type == "binary":
        if isinstance(criterion, nn.BCEWithlogits):
            all_preds, all_labels = bce_with_logits_inference(
                outputs,
                labels,
                all_preds,
                all_labels
            )
        elif isinstance(criterion, nn.BCE):
            all_preds, all_labels = bce_inference(
                outputs,
                labels,
                all_preds,
                all_labels
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
    else:
        raise NotImplementedError


def bce_with_logits_inference(outputs, labels, all_preds, all_labels):
    sigmoid = torch.sigmoid(outputs)
    predicted = sigmoid.data >= self.class_threshold
    for j in range(len(predicted)):
        all_preds.append(predicted[j].cpu().numpy()[0])
        all_labels.append(labels[j].cpu().numpy()[0])
    return all_preds, all_labels

def bce_inference(outputs, labels, all_preds, all_labels):
    predicted = outputs.data >= self.class_threshold
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
