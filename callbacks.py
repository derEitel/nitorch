from copy import deepcopy

class ModelCheckpoint():
    def __init__(self, path, retain_metric, mode, ignore_before):
        self.path = path
        self.retain_metric = retain_metric
        self.mode = mode
        self.ignore_before = ignore_before
        self.best_res = -1
        self.best_model = None
        
    def __call__(self, epoch, model, val_metrics):
        if epoch >= self.ignore_before:
            current_res = val_metrics[self.retain_metric.__name__][-1]
            if self.compare(current_res):
                self.best_res = current_res
                self.best_model = deepcopy(model.state_dict())

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
        """ Resets after training. Useful for cross validation """
        self.best_res = -1
        self.best_model = None
            
    def final(self):
        print("Best validation {} at {} after training.".format(self.retain_metric.__name__, self.best_res))
        # get best model and reset parameters
        best_model = deepcopy(self.model.state_dict())
        self.reset()
        return best_model