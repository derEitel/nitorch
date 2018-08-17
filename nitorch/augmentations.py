import numpy as np
import torch
from scipy.ndimage.interpolation import rotate


# Data augmentations
class SagittalFlip:
    def __call__(self, image):
        """ 
            Expects shape (X, Y, Z).
            Flips along the X axis (sagittal).
        """
        thresh = 0.5
        rand = np.random.uniform()
        if rand > thresh:
            augmented = np.flip(image, axis=0).copy()
        else:
            augmented = image
        return augmented


class Rotate:
    def __call__(self, image):
        """ 
            Expects shape (X, Y, Z).
            Rotates along the X axis.
        """
        min_rot, max_rot = -3, 3
        rand = np.random.randint(min_rot, max_rot + 1)
        augmented = rotate(
            image,
            angle=rand,
            axes=(1, 0),
            reshape=False
            ).copy()
        return augmented


class Translate:
    def __call__(self, image):
        """ 
            Expects shape (X, Y, Z).
            Translates the X axis.
        """
        min_trans, max_trans = -3, 3
        rand = np.random.randint(min_trans, max_trans + 1)
        augmented = np.zeros_like(image)
        if rand < 0:
            augmented[-rand:, :] = image[:rand, :]
        elif rand > 0:
            augmented[:-rand, :] = image[rand:, :]
        else:
            augmented = image
        return augmented


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    Expects labels to be scalar i.e. not tensors.
    """

    def __call__(self, image):
        # Expand with channel axis
        # numpy image: H x W x Z
        # torch image: C x H x W x Z

        image = torch.from_numpy(image).unsqueeze(0)
        image = image.float()
        return image
