import time
import numpy as np
import torch
from torch.autograd import Variable
from torch import nn
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from nitorch.inference import predict
from nitorch.callbacks import ModelCheckpoint


class Trainer:
    def __init__(
        self, 
        model, 
        criterion,
        optimizer,
        scheduler=None,
        metrics=[], 
        callbacks=[],
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
            callbacks: list of callbacks to execute. Default is None.
            class_threshold: classification threshold for binary 
                classification. Default is 0.5.
            prediction_type: accepts one of ["binary", "classification",
                "regression", "reconstruction", "other"]. This is used
                to determine output.

        """
        if not isinstance(model, nn.Module):
            raise ValueError('Expects model type to be torch.nn.Module')
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.metrics = metrics
        self.callbacks = callbacks
        self.class_threshold = None
        if "class_threshold" in kwargs.keys():
            self.class_threshold = kwargs["class_threshold"]
        self.stop_training = False
        self.start_time = None
        self.prediction_type = prediction_type

    def train_model(
        self,
        train_loader,
        val_loader,
        gpu=0,
        num_epochs=25,
        show_train_steps=25,
        show_validation_epochs=1,
        ):
        """ Main function to train a network."""

        val_metrics = dict()
        train_metrics = dict()

        self.start_time = time.time()
        self.best_metric = 0.0
        self.best_model = None
        
        for epoch in range(num_epochs):
            if self.stop_training:
                # TODO: check position of this
                print("Early stopping in epoch {}".format(epoch - 1))
                return self.finish_training(train_metrics, val_metrics)
            else:
                running_loss = 0.0
                epoch_loss = 0.0
                if self.scheduler:
                    self.scheduler.step(epoch)
                # variables to compute metrics
                all_preds = []
                all_labels = []
                # train
                self.model.train()
                for i, data in enumerate(train_loader):
                    try:
                        inputs, labels = data["image"], data["label"]
                    except TypeError:
                        # if data does not come in dictionary, assume
                        # that data is ordered like [input, label]
                        try:
                            inputs, labels = data[0], data[1]
                        except TypeError:
                            raise TypeError
                    # wrap data in Variable
                    inputs = Variable(inputs.cuda(gpu))
                    labels = Variable(labels.cuda(gpu))

                    # zero the parameter gradients
                    self.optimizer.zero_grad()
                    # forward + backward + optimize
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, labels)  
                    loss.backward()
                    self.optimizer.step()

                    # store results
                    all_preds, all_labels = predict(
                                outputs,
                                labels,
                                all_preds,
                                all_labels,
                                self.prediction_type,
                                self.criterion,
                                class_threshold=self.class_threshold
                            )
                    # print statistics
                    running_loss += loss.item()
                    epoch_loss += loss.item()
                    # print every X mini-batches
                    if i % show_train_steps == 0:  
                        print(
                            "[%d, %5d] loss: %.3f"
                            % (epoch + 1, i + 1, 
                               running_loss / show_train_steps)
                        )
                        running_loss = 0.0

                # report training metrics after each epoch
                train_metrics = self.report_metrics(
                    train_metrics,
                    self.metrics,
                    all_labels,
                    all_preds,
                    phase="Train"
                )

                epoch_loss /= len(train_loader)

                # add loss to metrics data
                if "loss" in train_metrics:
                    train_metrics["loss"].append(epoch_loss)
                else:
                    train_metrics["loss"] = [epoch_loss]

            # validate every x iterations
            if epoch % show_validation_epochs == 0:
                self.model.eval()
                validation_loss = 0.0
                all_preds = []
                all_labels = []

                with torch.no_grad():
                    for i, data in enumerate(val_loader):
                        try:
                            inputs, labels = data["image"], data["label"]
                        except TypeError:
                            # if data does not come in dictionary, assume
                            # that data is ordered like [input, label]
                            try:
                                inputs, labels = data[0], data[1]
                            except TypeError:
                                raise TypeError("Data not in correct \
                                 sequence format.")
                        # wrap data in Variable
                        inputs = Variable(inputs.cuda(gpu))
                        labels = Variable(labels.cuda(gpu))

                        # forward pass only
                        outputs = self.model(inputs)
                        loss = self.criterion(outputs, labels)
                        # compute validation accuracy
                        all_preds, all_labels = predict(
                                outputs,
                                labels,
                                all_preds,
                                all_labels,
                                self.prediction_type,
                                self.criterion,
                                class_threshold=self.class_threshold
                            )

                        validation_loss += loss.item()

                    val_metrics = self.report_metrics(
                        val_metrics,
                        self.metrics,
                        all_labels,
                        all_preds,
                        phase="Val"
                    )
                    validation_loss /= len(val_loader)
                    print("Val loss: {0:.4f}".format(validation_loss))
                    # add loss to metrics data
                    if "loss" in val_metrics:
                        val_metrics["loss"].append(validation_loss)
                    else:
                        val_metrics["loss"] = [validation_loss]
            if self.callbacks is not None:
                for callback in self.callbacks:
                    callback(self, epoch, val_metrics)
        # End training
        return self.finish_training(train_metrics, val_metrics)
        

    def finish_training(self, train_metrics, val_metrics):
        total_time = (time.time() - self.start_time) / 60
        print("Time trained: {:.2f} minutes".format(total_time))
        # execute final methods of callbacks
        if self.callbacks is not None:
            for callback in self.callbacks:
                # find all methods of the callback
                method_list = [
                    func
                    for func in dir(callback)
                    if (callable(getattr(callback, func)) 
                        and not func.startswith("__"))
                ]
                if "final" in method_list:
                    # in case of model checkpoint load best model
                    if isinstance(callback, ModelCheckpoint):
                        self.best_metric, best_model = callback.final()
                        self.model.load_state_dict(best_model)
                    else:
                        callback.final()
        # in case of no model selection, pick the last loss
        if self.best_metric == 0.0:
            self.best_metric = val_metrics["loss"][-1]
            self.best_model = self.model

        return (self.model, 
                {
                 "train_metrics" : train_metrics,
                 "val_metrics" : val_metrics,
                 "best_model" : self.best_model,
                 "best_metric" : self.best_metric}
                )



    def visualize_training(self, report, metrics=None):
        # Plot loss first
        plt.figure()
        plt.plot(report["train_metrics"]["loss"])
        plt.plot(report["val_metrics"]["loss"])
        plt.title("Loss during training")
        plt.legend(["Train", "Val"])
        plt.show()
        if metrics is None:
            metrics = self.metrics
        for metric in metrics:
            plt.figure()
            plt.plot(report["train_metrics"][metric.__name__])
            plt.plot(report["val_metrics"][metric.__name__])
            plt.legend(["Train", "Val"])
            plt.title(metric.__name__)
            plt.show()


    def evaluate_model(self, val_loader, gpu=0):
        # predict on the validation set
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data["image"], data["label"]
                # wrap data in Variable
                inputs = Variable(inputs.cuda(gpu))
                labels = Variable(labels.cuda(gpu))
                # forward + backward + optimize
                outputs = self.model(inputs)
                # run inference
                all_preds, all_labels = predict(
                                outputs,
                                labels,
                                all_preds,
                                all_labels,
                                self.prediction_type,
                                self.criterion,
                                class_threshold=self.class_threshold
                            )

        # compute confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
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
        plt.show()


    def report_metrics(
        self,
        metrics_dict,
        metrics,
        all_labels,
        all_preds, phase
        ):
        """ Function to compute a list of metric functions. """
        for metric in metrics:
            # report everything but loss
            if metric.__name__ is not "loss": 
                result = metric(all_labels, all_preds)
                if metric.__name__ in metrics_dict:
                    metrics_dict[metric.__name__].append(result)
                else:
                    metrics_dict[metric.__name__] = [result]
                # print result
                if isinstance(result, float):
                    print("{} {}: {:.2f} %".format(
                        phase, metric.__name__, result * 100))
                else:
                    print("{} {}: {} ".format(
                        phase, metric.__name__, str(result)))
        return metrics_dict
