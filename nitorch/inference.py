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
    if isinstance(prediction_type, list):
        # in case of multi-head classification separate
        output_names = kwargs["output_names"]
        if isinstance(all_preds, list):
            # convert to dictionary for multiple outputs
            all_preds = dict()
            all_labels = dict()
            for name in output_names:
                all_preds[name] = []
                all_labels[name] = []
            
        for pred_idx, pred_type in enumerate(prediction_type):
            all_preds[output_names[pred_idx]], all_labels[output_names[pred_idx]]  = predict(
                outputs=outputs[pred_idx],
                labels=labels[pred_idx],
                all_preds=all_preds[output_names[pred_idx]],
                all_labels=all_labels[output_names[pred_idx]],
                criterion=criterion,
                prediction_type=pred_type,
                output_name=output_names[pred_idx],
                **kwargs
            )
        return all_preds, all_labels

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
    if prediction_type == "binary_with_logits":
        all_preds, all_labels = bce_with_logits_inference(
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
                all_labels,
                **kwargs
        )
        return all_preds, all_labels
    elif prediction_type == "reconstruction":
        # TODO: test different loss functions
        all_preds, all_labels = regression_inference(
                outputs,
                labels,
                all_preds,
                all_labels,
                **kwargs
        )
        return all_preds, all_labels
    elif prediction_type == "variational":
        # TODO: test different loss functions
        all_preds, all_labels = variational_inference(
                outputs,
                labels,
                all_preds,
                all_labels,
                **kwargs
        )
        return all_preds, all_labels
    else:
        raise NotImplementedError


def bce_with_logits_inference(
    outputs,
    labels,
    all_preds,
    all_labels,
    **kwargs
    ):
    sigmoid = torch.sigmoid(outputs)
    class_threshold = 0.5
    if "class_threshold" in kwargs.keys():
        if kwargs["class_threshold"] is not None:
            class_threshold = kwargs["class_threshold"]

    predicted = sigmoid.data >= class_threshold
    for j in range(len(predicted)):
        #all_preds.append(predicted[j].cpu().numpy()[0])
        #all_labels.append(labels[j].cpu().numpy()[0])
        all_preds.append(predicted[j].cpu().numpy().item())
        all_labels.append(labels[j].cpu().numpy().item())
    return all_preds, all_labels

def bce_inference(
    outputs,
    labels,
    all_preds,
    all_labels,
    **kwargs
    ):
    if "class_threshold" in kwargs.keys():
        class_threshold = kwargs["class_threshold"]
    else:
        class_threshold = 0.5
    predicted = outputs.data >= class_threshold
    for j in range(len(predicted)):
        all_preds.append(float(predicted[j].cpu().numpy()[0]))
        all_labels.append(float(labels[j].cpu().numpy()[0]))
    return all_preds, all_labels

def regression_inference(
    outputs,
    labels,
    all_preds,
    all_labels,
    **kwargs
    ):
    # Multi-head case
    # network returns a tuple of outputs
    if isinstance(outputs, tuple):
        predicted = [output.data for output in outputs]
        for head in range(len(predicted)):
            for j in range(len(predicted[head])):
                try:
                    all_preds[head].append(predicted[head][j].cpu().numpy()[0])
                    all_labels[head].append(labels[head][j].cpu().numpy()[0])
                except IndexError:
                    # create inner lists if needed
                    all_preds.append([predicted[head][j].cpu().numpy()[0]])
                    all_labels.append([labels[head][j].cpu().numpy()[0]])
        return all_preds, all_labels
    # Single-head case
    else:
        predicted = outputs.data
        # TODO: replace for loop with something faster
        for j in range(len(predicted)):
            #try:
                #all_preds.append(predicted[j].cpu().numpy().item())
                #all_labels.append(labels[j].cpu().numpy().item())
            #except:

            all_preds.append(float(predicted[j].cpu().item()))
            all_labels.append(float(labels[j].cpu().item()))
        return all_preds, all_labels

def variational_inference(
    outputs,
    labels,
    all_preds,
    all_labels,
    **kwargs
    ):
    """ Inference for variational autoencoders. """
    # VAE outputs reconstruction, mu and std
    # select reconstruction only
    outputs = outputs[0]
    predicted = outputs.data 
    # TODO: replace for loop with something faster
    for j in range(len(predicted)):
        all_preds.append(predicted[j].cpu().numpy()[0])
        all_labels.append(labels[j].cpu().numpy()[0])
    return all_preds, all_labels
