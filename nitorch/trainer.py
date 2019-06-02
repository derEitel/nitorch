import time
import numpy as np
import torch
from torch import nn
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
from nitorch.inference import predict
from nitorch.callbacks import ModelCheckpoint
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
            device=torch.device('cuda'),
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
        self.prediction_type = prediction_type


    def train_model(
            self,
            train_loader,
            val_loader,
            inputs_key="image",
            labels_key="label",
            num_epochs=25,
            show_train_steps=None,
            show_validation_epochs=1
    ):
        """ Main function to train a network for one epoch.
        Args:
            train_loader: a pytorch Dataset iterator for training data
            val_loader: a pytorch Dataset iterator for validation data
            inputs_key, labels_key: The data returned by `train_loader` and `val_loader`can
                            either be a dict of format data_loader[X_key] = inputs and
                            data_loader[y_key] = labels or a list with data_loader[0] = inputs
                            and data_loader[1] = labels. The default keys are "image" and "label".
        """
        n = len(train_loader)
        n_val = len(val_loader)
        # if show_train_steps is not specified then default it to print training progress 4 times per epoch
        if not(show_train_steps):
            show_train_steps = n//4 if((n//4)>1) else 1
        store_val_steps = n_val//4 if((n_val//4)>1) else 1

        assert (show_train_steps>0) and (show_train_steps<=n),"\
'show_train_steps' value-{} is out of range. Must be >0 and <={} i.e. len(train_loader)".format(show_train_steps, n)
        assert (show_validation_epochs < num_epochs) or (num_epochs == 1), "\
'show_validation_epochs' value should be less than 'num_epochs'"
        
        # store metrics and loss for each epoch to report in the end
        self.val_metrics = {"loss":[]}
        self.train_metrics = {"loss":[]}
        if(self.metrics):
            self.val_metrics.update({m.__name__:[] for m in self.metrics})
            self.train_metrics.update({m.__name__:[] for m in self.metrics})

        self.start_time = time.time()
        self.best_metric = 0.0
        self.best_model = None

        for epoch in range(num_epochs):
            if self.stop_training:
                # TODO: check position of this
                print("Early stopping in epoch {}".format(epoch))
                return self.finish_training(epoch)
            else:
                # 'running_loss' accumulates loss until it gets printed every 'show_train_steps'.
                running_loss = []
                # 'accumulates predictions and labels until the metrics are calculated in the epoch
                all_outputs = torch.Tensor().to(self.device)
                all_labels = torch.Tensor().to(self.device)

                if self.scheduler:
                    self.scheduler.step(epoch)

                # train mode
                self.model.train()

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

                    if self.training_time_callback is not None:
                        outputs = self.training_time_callback(
                            inputs, labels, i, epoch)
                    else:
                        # forward + backward + optimize
                        outputs = self.model(inputs)

                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    # enable the below commented code if you want to visualize the 
                    # gradient flow through the model during training
                    # plot_grad_flow(self.model.named_parameters())
                    self.optimizer.step()

                    # update loss
                    running_loss.append(loss.item())
                    # print loss every 'show_train_steps' mini-batches
                    if (i != 0) and (i % show_train_steps == 0):
                        print(
                            "[%d, %5d] loss: %.5f"
                            % (epoch , i , np.mean(running_loss))
                        )

                    # store the outputs and labels for computing metrics later
                    if(self.prediction_type == "reconstruction"):
                    # when output/label tensors are very large (e.g. for reconstruction tasks) 
                    # store the outputs/labels only a few times
                        if(i % show_train_steps == 0):
                            all_outputs = torch.cat((all_outputs, outputs.float()))
                            all_labels = torch.cat((all_labels, labels.float()))
                    else:
                        all_outputs = torch.cat((all_outputs, outputs.float()))
                        all_labels = torch.cat((all_labels, labels.float()))

                #<end-of-training-cycle-loop>
            #<end-of-epoch-loop>
            # at the end of an epoch, calculate metrics, report them and
            # store them in respective report dicts
            self._estimate_metrics(all_outputs, all_labels, np.mean(running_loss), phase="train")

            # validate every x iterations
            if(epoch % show_validation_epochs == 0):
                running_loss_val = []
                all_outputs = torch.Tensor().to(self.device)
                all_labels = torch.Tensor().to(self.device)

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
                    if(self.prediction_type == "reconstruction"):
                        if(i % store_val_steps == 0):
                            all_outputs = torch.cat((all_outputs, outputs.float()))
                            all_labels = torch.cat((all_labels, labels.float()))
                    else:
                        all_outputs = torch.cat((all_outputs, outputs.float()))
                        all_labels = torch.cat((all_labels, labels.float()))

                    validation_loss = np.mean(running_loss_val)
                    print("val loss: {0:.6f}".format(validation_loss))
                    # add loss to metrics data

                # report validation metrics
                # weighted averages of metrics are computed over batches
                self._estimate_metrics(all_outputs, all_labels, validation_loss, phase="val")

            if self.callbacks:
                for callback in self.callbacks:
                    callback(self, epoch, self.val_metrics)
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
                    callback.final(trainer=self, epoch=epoch)
        # in case of no model selection, pick the last loss
        if self.best_metric == 0.0:
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
        # Plot loss first
        plt.figure()
        plt.plot(report["train_metrics"]["loss"])
        plt.plot(report["val_metrics"]["loss"])
        plt.title("Loss during training")
        plt.legend(["Train", "Val"])
        if (save_fig_path):
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
            if(save_fig_path):
                plt.savefig(save_fig_path+"_"+metric.__name__)
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

        all_outputs = torch.Tensor().to(device)
        all_labels = torch.Tensor().to(device)

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data[inputs_key], data[labels_key]
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = self.model(inputs)
                all_outputs = torch.cat((all_outputs, outputs.float()))
                all_labels = torch.cat((all_labels, labels.float()))

            # run inference
            all_preds = predict(
                all_outputs,
                self.prediction_type,
                self.criterion,
                class_threshold=self.class_threshold
            )

        # calculate the loss criterion metric
        loss_score = self.criterion(all_outputs, all_labels)
        results = {"loss":loss_score.item()}

        # calculate metrics
        for metric in metrics:                
            if isinstance(all_preds[0], list):
                score = np.mean([metric(preds, labels) for preds,labels in zip(all_preds, all_labels)])
            else:
                score = metric(all_preds, all_labels)

            results.update({metric.__name__:score})

        if(write_to_dir):
            with open(write_to_dir+"results.json","w") as f:
                json.dump(results, f)

        print("evaluation results :\n",results)

        # compute confusion matrix if it is a binary classification task
        if(self.prediction_type == "binary"):
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
            if(write_to_dir):
                plt.savefig(write_to_dir+"confusion_matrix.png")
            else:
                plt.show()


        self.model.train()


    def _report_metrics(
        self,
        metrics_dict,
        phase
        ):

        # report execution time only in training phase
        if (phase == "train"):
            time_elapsed = int(time.time() - self.start_time)
            print("Time elapsed: {}h:{}m:{}s".format(
                time_elapsed // 3600, (time_elapsed // 60) % 60, time_elapsed % 60))

        # print the scores
        for metric in self.metrics:
            score = metrics_dict[metric.__name__][-1]
            if isinstance(score, float):
                print("{} {}: {:.2f} %".format(
                    phase, metric.__name__, score * 100))
            else:
                print("{} {}: {} ".format(
                    phase, metric.__name__, str(score)))


    def _estimate_metrics(
        self,
        all_outputs,
        all_labels,
        loss,
        phase
        ):
        '''at the end of an epoch 
        (a) calculate metrics 
        (b) report metrics 
        (c) store results in respective report dicts - train_metrics / val_metrics '''
        # print("<_estimate_metrics>", phase, all_outputs.shape, "loss", loss.shape)
        # print("<train_metrics>", self.train_metrics)
        # print("<val_metrics>", self.val_metrics)
        if(phase.lower() == 'train'):
            metrics_dict = self.train_metrics
        elif(phase.lower() == 'val'):
            metrics_dict = self.val_metrics
        else:
            assert "Invalid 'phase' defined. Can only be one of ['train', 'val']"

        # add loss to metrics_dict
        metrics_dict["loss"].append(loss)
        # add other metrics to the metrics_dict
        if (all_outputs.nelement()): # check for unreported metrics
            # perform inference on the outputs
            all_preds = predict(all_outputs
                , self.prediction_type
                , self.criterion
                , class_threshold=self.class_threshold
                )

            for metric in self.metrics:
                if isinstance(all_preds[0], list):
                    result = np.mean([metric(labels, preds) for preds,labels in zip(all_preds, all_labels)])
                else:
                    result = metric(all_labels, all_preds)

                metrics_dict[metric.__name__].append(result)

        self._report_metrics(metrics_dict, phase)
        