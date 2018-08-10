from copy import deepcopy


class ModelCheckpoint:
    def __init__(self, path, retain_metric, mode, ignore_before=0):

        # TODO: implement saving
        self.path = path
        self.retain_metric = retain_metric
        self.mode = mode
        self.ignore_before = ignore_before
        self.best_res = -1
        self.best_model = None

    def __call__(self, trainer, epoch, val_metrics):
        if epoch >= self.ignore_before:
            if isinstance(self.retain_metric, str):
                current_res = val_metrics[self.retain_metric][-1]
            else:
                current_res = val_metrics[self.retain_metric.__name__][-1]
            if self.compare(current_res):
                self.best_res = current_res
                self.best_model = deepcopy(trainer.model.state_dict())

    def compare(self, res):
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

    def reset(self):
        """
        Reset module after training.
        Useful for cross validation.
        """
        self.best_res = -1
        self.best_model = None

    def final(self):
        """ get best model and reset parameters."""
        if isinstance(self.retain_metric, str):
            name = self.retain_metric
        else:
            name = self.retain_metric.__name__
        print(
            "Best validation {} at {} after training.".format(
                name, self.best_res
            )
        )
        if self.best_model is not None:
            best_model = deepcopy(self.best_model)
            best_res = self.best_res
            self.reset()
            return best_res, best_model
        else:
            print("Minimum iterations to store model not reached.")
            self.reset()
            return self.best_res, best_model


class EarlyStopping:
    """ Stop training when a monitored quantity has stopped improving.

    # Arguments
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
                if self.compare(current_res):
                    self.best_res = current_res
                    self.best_epoch = epoch
            else:
                # end training run
                trainer.stop_training = True

    def compare(self, res):
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

    def final(self):
        self.reset()