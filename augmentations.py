import numpy as np
import torch
from scipy.ndimage.interpolation import rotate


# Data augmentations
class SagittalFlip:
    def __call__(self, batch):
        """ 
            Expects shape (X, Y, Z).
            Flips along the X axis (sagittal).
        """
        images, labels = batch["image"], batch["label"]
        thresh = 0.5
        rand = np.random.uniform()
        if rand > thresh:
            batch_augmented = np.flip(images, axis=0).copy()
        else:
            batch_augmented = images
        return {"image": batch_augmented, "label": labels}


class Rotate:
    def __call__(self, batch):
        """ 
            Expects shape (X, Y, Z).
            Rotates along the X axis.
        """
        images, labels = batch["image"], batch["label"]
        min_rot, max_rot = -3, 3
        rand = np.random.randint(min_rot, max_rot + 1)
        batch_augmented = rotate(images, angle=rand, axes=(1, 0), reshape=False).copy()
        return {"image": batch_augmented, "label": labels}


class Translate:
    def __call__(self, batch):
        """ 
            Expects shape (X, Y, Z).
            Translates the X axis.
        """
        images, labels = batch["image"], batch["label"]
        min_trans, max_trans = -3, 3
        rand = np.random.randint(min_trans, max_trans + 1)
        batch_augmented = np.zeros_like(images)
        if rand < 0:
            batch_augmented[-rand:, :] = images[:rand, :]
        elif rand > 0:
            batch_augmented[:-rand, :] = images[rand:, :]
        else:
            batch_augmented = images
        return {"image": batch_augmented, "label": labels}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, batch):
        images, labels = batch["image"], batch["label"]
        # Expand with channel axis
        # numpy image: H x W x Z
        # torch image: C x H x W x Z

        images = torch.from_numpy(images).unsqueeze(0)
        images = images.float()
        labels = torch.FloatTensor([labels])

        return {"image": images, "label": labels}
