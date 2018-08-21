import torch

def dataset_length(data_loader):
    """
    Return the full length of the dataset from the DataLoader alone.
    Calling len(data_loader) only shows the number of mini-batches.
    Requires data to be located at 
    """
    sample = next(iter(data_loader))

    if isinstance(sample, dict):
        try:
            if isinstance(sample["label"], torch.Tensor):
                batch_size = sample["label"].shape[0]
            else:
                # in case of sequence of inputs use first input
                batch_size = sample["label"][0].shape[0]
        except:
            KeyError("Expects key to be 'label'.")
    else:
        if isinstance(sample[1], torch.Tensor):
            batch_size = sample[1].shape[0]
        else:
            # in case of sequence of inputs use first input
            batch_size = sample[1][0].shape[0]
    return len(data_loader) * batch_size