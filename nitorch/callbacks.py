import os
import copy
import torch
from copy import deepcopy

class Callback:
    """
    Abstract class for callbacks.
    """

    def __init__(self):
        pass

    def __call__(self):
        pass

    def reset(self):
        pass

    def final(self, **kwargs):
        self.reset()

class ModelCheckpoint(Callback):
    """
    # TODO

    Arguments:
        path:
        num_iters: number of iterations after which to store the model.
            If set to -1, it will only store the last iteration's model.
        prepend: string to prepend the filename with.
        ignore_before: ignore early iterations.
        store_best: boolen whether to save the best model during
            training.
        store_best_metric: name of the metric to use for best model
            selection.
        mode: "max" or "min".
    """

    def __init__(
        self,
        path,
        prepend="",
        num_iters=-1,
        ignore_before=0,
        store_best=False,
        retain_metric="accuracy_score",
        mode="max"
        ):
        super().__init__()
        if os.path.isdir(path):
            self.path = path
        else:
            os.makedirs(path)
            self.path = path
        # end the prepended text with an underscore if it does not
        if not prepend.endswith("_") and prepend != "":
            prepend += "_"
        self.prepend = prepend
        self.num_iters = num_iters
        self.ignore_before = ignore_before
        self.best_model = None
        self.best_res = -1
        self.store_best = store_best
        self.retain_metric = retain_metric
        self.mode = mode

    def __call__(self, trainer, epoch, val_metrics):
        # do not store intermediate iterations
        if epoch >= self.ignore_before and epoch != 0:
            if not self.num_iters == -1:
            
                # counting epochs starts from 1; i.e. +1
                epoch += 1
                # store model recurrently if set
                if epoch % self.num_iters == 0:
                    name = self.prepend + "training_epoch_{}.h5".format(epoch)
                    full_path = os.path.join(self.path, name)
                    self.save_model(trainer, full_path)

            # store current model if improvement detected
            if self.store_best:
                current_res = 0
                # use loss directly
                if self.retain_metric == "loss":
                    curent_res = val_metrics["loss"][-1]
                else: 
                    try:
                        # check if value can be used directly or not
                        if isinstance(self.retain_metric, str):
                            current_res = val_metrics[self.retain_metric][-1]
                        else:
                            current_res = val_metrics[self.retain_metric.__name__][-1]
                    except KeyError:
                        print("Couldn't find {} in validation metrics. Using \
                            loss instead.".format(retain_metric))
                        curent_res = val_metrics["loss"][-1]
                if self.has_improved(current_res):
                    self.best_res = current_res
                    self.best_model = deepcopy(trainer.model.state_dict())

    def reset(self):
        """
        Reset module after training.
        Useful for cross validation.
        """
        self.best_model = None
        self.best_res = -1

    def final(self, **kwargs):
        epoch = kwargs["epoch"] + 1
        if epoch >= self.ignore_before:
            name = self.prepend + "training_epoch_{}_FINAL.h5".format(epoch)
            full_path = os.path.join(self.path, name)
            self.save_model(kwargs["trainer"], full_path)
        else:
            print("Minimum iterations to store model not reached.")

        if self.best_model is not None:
            best_model = deepcopy(self.best_model)
            best_res = self.best_res
            print("Best result during training: {}. Saving model..".format(best_res))
            name = self.prepend + "BEST_ITERATION.h5"
            torch.save(best_model, os.path.join(self.path, name))
        self.reset()

    def save_model(self, trainer, full_path):
        print("Writing model to disk...")
        model = trainer.model.cpu()
        torch.save(model.state_dict(), full_path)
        if trainer.device is not None:
            trainer.model.cuda(trainer.device)

    def has_improved(self, res):
        if self.mode == "max":
            return res >= self.best_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_res == -1:
                return True
            else:
                return res <= self.best_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")


class EarlyStopping(Callback):
    """ 
    Stop training when a monitored quantity has stopped improving.

    Arguments
        patience: number of iterations without improvement after which
            to stop
        retain_metric: the metric which you want to monitor
        mode: {min or max}; defines if you want to maximise or minimise
            your metric
        ignore_before: does not start the first window until this epoch.
            Can be useful when training spikes a lot in early epochs.
    """


    def __init__(self, patience, retain_metric, mode, ignore_before=0):
        self.patience = patience
        self.retain_metric = retain_metric
        self.mode = mode
        self.ignore_before = ignore_before
        self.best_res = -1
        # set to first iteration which is interesting
        self.best_epoch = self.ignore_before

    def __call__(self, trainer, epoch, val_metrics):
        if epoch >= self.ignore_before:
            if epoch - self.best_epoch < self.patience:
                if isinstance(self.retain_metric, str):
                    current_res = val_metrics[self.retain_metric][-1]
                else:
                    current_res = val_metrics[self.retain_metric.__name__][-1]
                if self.has_improved(current_res):
                    self.best_res = current_res
                    self.best_epoch = epoch
            else:
                # end training run
                trainer.stop_training = True

    def has_improved(self, res):
        if self.mode == "max":
            return res > self.best_res
        elif self.mode == "min":
            # check if still standard value
            if self.best_res == -1:
                return True
            else:
                return res < self.best_res
        else:
            raise NotImplementedError("Only modes 'min' and 'max' available")

    def reset(self):
        """ Resets after training. Useful for cross validation."""
        self.best_res = -1
        self.best_epoch = self.ignore_before

    def final(self, **kwargs):
        self.reset()
