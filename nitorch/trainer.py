import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from nitorch.inference import predict
from nitorch.utils import *
import json


class Trainer:
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler=None,
            metrics=[],
            callbacks=[],
            training_time_callback=None,
            device=torch.device("cuda"),
            prediction_type="binary",
            **kwargs
    ):
        """ Main class for training.
        # Arguments
            model: neural network to train.
            criterion: loss function.
            optimizer: optimization function.
            scheduler: schedules the optimizer.
            metrics: list of metrics to report. Default is None.
            callbacks: list of callbacks to execute at the end of training epochs. Default is None.
            training_time_callback: a user-defined callback that executes the model.forward()
                and returns the output to the trainer.
                This can be used to perform debug during train time, Visualize features,
                call model.forward() with custom arguments, run multiple decoder networks etc.
                Default is None.
            class_threshold: classification threshold for binary
                classification. Default is 0.5.
            prediction_type: accepts one of ["binary", "classification",
                "regression", "reconstruction", "variational", "other"].
                This is used to determine output type.
            device: The device to use for training. Must be integer or
                    a torch.device object. By default, GPU with current
                    node is used.
        """
        if not isinstance(model, nn.Module):
            raise ValueError("Expects model type to be torch.nn.Module")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.callbacks = callbacks
        self.training_time_callback = training_time_callback
        if isinstance(device, int):
            self.device = torch.device("cuda:" + str(device))
        elif isinstance(device, torch.device):
            self.device = device
        else:
            raise ValueError("Device needs to be of type torch.device or \
                integer.")
        if "class_threshold" in kwargs.keys():
            self.class_threshold = kwargs["class_threshold"]
        else:
            self.class_threshold = None
        self.stop_training = False
        self.start_time = None
        self.prediction_type = prediction_type #TODO : for multi-head tasks allow a list of prediction_types to be provided
        self.val_metrics = {"loss": []}
        self.train_metrics = {"loss": []}
        self.best_metric = None
        self.best_model = None

    def train_model(
            self,
            train_loader,
            val_loader,
            inputs_key="image",
            labels_key="label",
            num_epochs=25,
            show_train_steps=None,
            show_validation_epochs=1,
            store_grads=False
    ):
        """ Main function to train a network for one epoch.
        Args:
            train_loader: a pytorch Dataset iterator for training data
            val_loader: a pytorch Dataset iterator for validation data
            inputs_key, labels_key: The data returned by `train_loader` and `val_loader`can
                            either be a dict of format data_loader[X_key] = inputs and
                            data_loader[y_key] = labels or a list with data_loader[0] = inputs
                            and data_loader[1] = labels. The default keys are "image" and "label".
            store_grads (optional): allows visualization of the gradient flow through the model during training.
            After calling this method, do plt.show() to see the gradient flow diagram.
        """
        n = len(train_loader)
        n_val = len(val_loader)
        # if show_train_steps is not specified then default it to print training progress 4 times per epoch
        if not show_train_steps:
            show_train_steps = n // 4 if ((n // 4) > 1) else 1

        assert (show_train_steps > 0) and (show_train_steps <= n), "\
'show_train_steps' value-{} is out of range. Must be >0 and <={} i.e. len(train_loader)".format(show_train_steps, n)
        assert (show_validation_epochs < num_epochs) or (num_epochs == 1), "\
'show_validation_epochs' value should be less than 'num_epochs'"

        # reset metric dicts
        self.val_metrics = {"loss": []}
        self.train_metrics = {"loss": []}

        self.start_time = time.time()
        self.best_metric = None
        self.best_model = None

        for epoch in range(num_epochs):
            # if early stopping is on, check if stop signal is switched on
            if self.stop_training:
                return self.finish_training(epoch)
            else:
                # train mode
                self.model.train()
                # 'running_loss' accumulates loss until it gets printed every 'show_train_steps'.
                # 'accumulates predictions and labels until the metrics are calculated in the epoch
                running_loss = []
                all_outputs = []
                all_labels = []

                if self.scheduler:
                    self.scheduler.step(epoch)
    
                for i, data in enumerate(train_loader):
                    try:
                        inputs, labels = data[inputs_key], data[labels_key]
                    except TypeError:
                        # if data does not come in dictionary, assume
                        # that data is ordered like [input, label]
                        try:
                            inputs, labels = data[0], data[1]
                        except TypeError:
                            raise TypeError
                    
                    # in case of multi-input or output create a list
                    if isinstance(inputs, list):
                        inputs = [inp.to(self.device) for inp in inputs]
                    else:
                        inputs = inputs.to(self.device)
                    if isinstance(labels, list):
                        labels = [label.to(self.device) for label in labels]
                    else:
                        labels = labels.to(self.device)

                    # zero the parameter gradients
                    self.optimizer.zero_grad()

                    if self.training_time_callback:
                        outputs = self.training_time_callback(
                            inputs, labels, i, epoch)
                    else:
                        # forward + backward + optimize
                        outputs = self.model(inputs)

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    
                    # update loss
                    running_loss.append(loss.item())
                    # print loss every 'show_train_steps' mini-batches
                    if(i % show_train_steps == 0):
                        if(i != 0):
                            print(
                                "[%d, %5d] loss: %.5f"
                                % (epoch, i, np.mean(running_loss))
                            )
                            
                        # store the outputs and labels for computing metrics later     
                        all_outputs.append(outputs)
                        all_labels.append(labels)
                        # allows visualization of the gradient flow through the model during training
                        if(store_grads):
                            plot_grad_flow(self.model.named_parameters())
                            
                # <end-of-training-cycle-loop>
                # at the end of an epoch, calculate metrics, report them and
                # store them in respective report dicts
                self._estimate_and_report_metrics(
                    all_outputs, all_labels, running_loss, 
                    metrics_dict=self.train_metrics, 
                    phase="train"
                )
                del all_outputs, all_labels, running_loss
                
                # validate every x iterations
                if epoch % show_validation_epochs == 0:
                    running_loss_val = []
                    all_outputs = []
                    all_labels = []

                    self.model.eval()

                    with torch.no_grad():
                        for i, data in enumerate(val_loader):
                            try:
                                inputs, labels = data[inputs_key], data[labels_key]
                            except TypeError:
                                # if data does not come in dictionary, assume
                                # that data is ordered like [input, label]
                                try:
                                    inputs, labels = data[0], data[1]
                                except TypeError:
                                    raise TypeError("Data not in correct \
                                     sequence format.")
                            # in case of multi-input or output create a list
                            if isinstance(inputs, list):
                                inputs = [inp.to(self.device) for inp in inputs]
                            else:
                                inputs = inputs.to(self.device)
                            if isinstance(labels, list):
                                labels = [label.to(self.device) for label in labels]
                            else:
                                labels = labels.to(self.device)

                            # forward pass only
                            if self.training_time_callback is not None:
                                outputs = self.training_time_callback(
                                    inputs,
                                    labels,
                                    1,  # dummy value
                                    1  # dummy value
                                )
                            else:
                                outputs = self.model(inputs)

                            loss = self.criterion(outputs, labels)

                            running_loss_val.append(loss.item())
                            # store the outputs and labels for computing metrics later
                            all_outputs.append(outputs)
                            all_labels.append(labels)

                    # report validation metrics
                    self._estimate_and_report_metrics(
                        all_outputs, all_labels, running_loss_val, 
                        metrics_dict=self.val_metrics, 
                        phase="val"
                    )
                    del all_outputs, all_labels, running_loss_val
                    
            # <end-of-epoch-loop>
            for callback in self.callbacks:
                callback(self, epoch=epoch)
        # End training
        return self.finish_training(epoch)

    def finish_training(self, epoch):
        """
        End the training cyle, return a model and finish callbacks.
        """
        time_elapsed = int(time.time() - self.start_time)
        print("Total time elapsed: {}h:{}m:{}s".format(
            time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

        # execute final methods of callbacks
        for callback in self.callbacks:
            # find all methods of the callback
            try:
                callback.final(trainer=self, epoch=epoch)
            except AttributeError:
                pass

        # in case of no model selection, pick the last loss
        if not self.best_metric:
            self.best_metric = self.val_metrics["loss"][-1]
            self.best_model = self.model

        return (self.model,
                {
                    "train_metrics": self.train_metrics,
                    "val_metrics": self.val_metrics,
                    "best_model": self.best_model,
                    "best_metric": self.best_metric}
                )

    def visualize_training(self, report, metrics=None, save_fig_path=""):
        # TODO plot metrics for all tasks in a multi-task training 
        # Plot loss first
        plt.figure()
        plt.plot(report["train_metrics"]["loss"])
        plt.plot(report["val_metrics"]["loss"])
        plt.title("Loss during training")
        plt.legend(["Train", "Val"])
        if save_fig_path:
            plt.savefig(save_fig_path)
        plt.show()
        if metrics is None:
            metrics = self.metrics
        for metric in metrics:
            plt.figure()
            plt.plot(report["train_metrics"][metric.__name__])
            plt.plot(report["val_metrics"][metric.__name__])
            plt.legend(["Train", "Val"])
            plt.title(metric.__name__)
            if save_fig_path:
                plt.savefig(save_fig_path + "_" + metric.__name__)
                plt.close()
            else:
                plt.show()

    def evaluate_model(
            self,
            val_loader,
            additional_gpu=None,
            metrics=[],
            inputs_key="image",
            labels_key="label",
            write_to_dir=''
    ):
        # predict on the validation set
        """
        Predict on the validation set.
        # Arguments
            val_loader : data loader of the validation set
            additional_gpu : GPU number if evaluation should be done on
                separate GPU
            metrics: list of
            write_to_dir: the outputs of the evaluation are written to files path provided
        """

        self.model.eval()

        if additional_gpu is not None:
            device = additional_gpu
        else:
            device = self.device
            
        running_loss = []
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data[inputs_key], data[labels_key]
                # in case of multi-input or output create a list
                if isinstance(inputs, list):
                    inputs = [inp.to(self.device) for inp in inputs]
                else:
                    inputs = inputs.to(self.device)
                if isinstance(labels, list):
                    labels = [label.to(self.device) for label in labels]
                else:
                    labels = labels.to(self.device)
                    
                if self.training_time_callback:
                    outputs = self.training_time_callback(
                        inputs, labels, 1, 1)
                else:
                    outputs = self.model(inputs)
                    
                loss = self.criterion(outputs, labels)
                
                running_loss.append(loss.item())
                all_outputs.append(outputs)
                all_labels.append(labels)
                
            # calculate the loss criterion metric
            results = {"loss": []}
                
            # calculate metrics
            self._estimate_and_report_metrics(
                all_outputs, all_labels, running_loss, 
                metrics_dict=results, 
                phase="eval"
            )

        if write_to_dir:
            results = {k:v[0] for k,v in results.items()}
            with open(write_to_dir + "results.json", "w") as f:
                json.dump(results, f)
            
            # compute confusion matrix if it is a binary classification task
            if self.prediction_type == 'binary':
                plt.savefig(write_to_dir + "confusion_matrix.png")
                plt.close()
        else:
            if self.prediction_type == 'binary':
                plt.show()

        self.model.train()

        
    def _estimate_and_report_metrics(
            self,
            all_outputs,
            all_labels,
            running_loss,
            metrics_dict,
            phase
    ):
        """
        at the end of an epoch
        (a) calculate metrics
        (b) store results in respective report dicts
        (c) report metrics """                        
        # report execution time, only in training phase
        if phase == "train":
            time_elapsed = int(time.time() - self.start_time)
            print("Time elapsed: {}h:{}m:{}s".format(
                time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))
            
        if isinstance(all_outputs[0], list):
            all_outputs = [torch.cat(out).float() for out in zip(*all_outputs)]
            all_labels = [torch.cat(lbl).float() for lbl in zip(*all_labels)]                   
        else:
            all_outputs = [torch.cat(all_outputs).float()]
            all_labels = [torch.cat(all_labels).float()]

        # add loss to metrics_dict
        loss = np.mean(running_loss)
        metrics_dict["loss"].append(loss)
        # print the loss for val and eval phases
        if phase != "train":
            print("{} loss: {:.5f}".format(phase, loss))
            
        # add other metrics to the metrics_dict
        for i, (all_output, all_label) in enumerate(zip(all_outputs, all_labels)):
            # perform inference on the outputs
            all_pred, all_label = predict(
                                all_output, 
                                all_label,
                                self.prediction_type,
                                self.criterion,
                                class_threshold=self.class_threshold
                                )
            # If it is a multi-head training then append prefix            
            if i==0:
                metric_prefix = "" 
            else:
                metric_prefix = "task{} ".format(i+1)
                
            # report metrics
            for metric in self.metrics:
                metric_name = metric_prefix + metric.__name__ 
                result = metric(all_label, all_pred)
                if isinstance(result, float):
                    print("{} {}: {:.2f} %".format(
                        phase, metric_name, result * 100))
                else:
                    print("{} {}: {} ".format(
                        phase, metric_name, str(result)))
                # store results in the report dict
                if metric_name in metrics_dict:
                    metrics_dict[metric_name].append(result)
                else:                    
                    metrics_dict[metric_name] = [result]
                
                # plot confusion graph if it is a binary classification
                if phase == "eval" and self.prediction_type == "binary":      
                    cm = confusion_matrix(all_label, all_pred)
                    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)

                    # Visualize the confusion matrix
                    classes = ["control", "patient"]
                    tick_marks = np.arange(len(classes))
                    plt.xticks(tick_marks, classes, rotation=45)
                    plt.yticks(tick_marks, classes)

                    fmt = "d"
                    thresh = cm.max() / 2.
                    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                        plt.text(
                            j,
                            i,
                            format(cm[i, j], fmt),
                            horizontalalignment="center",
                            color="white" if cm[i, j] > thresh else "black",
                        )
                    plt.title("Confusion Matrix")
                    plt.ylabel("True label")
                    plt.xlabel("Predicted label")