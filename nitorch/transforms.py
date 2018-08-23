import numpy as np
import torch
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import zoom


def normalize_float(x, min=-1):
    """ 
    Function that performs min-max normalization on a `numpy.ndarray` matrix. 
    """
    if min == -1:
        norm = (2 * (x - np.min(x)) / (np.max(x) - np.min(x))) - 1
    elif min == 0:
        if np.max(x) == 0 and np.min(x) == 0:
            norm = x
        else:
            norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    return norm


def normalize_float_torch(x, min=-1):
    '''
     Function that performs min-max normalization on a Pytorch tensor matrix.
     Can also deal with Pytorch dictionaries where the data matrix
     key is 'image'.
    '''
    import torch
    if min == -1:
        norm = (2 * (x - torch.min(x)) / (torch.max(x) - torch.min(x))) - 1
    elif min == 0:
        if torch.max(x) == 0 and torch.min(x) == 0:
            norm = x
        else:    
            norm = (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    return norm


def normalization_factors(data, train_idx, shape, mode="slice"):
    """ 
    Shape should be of length 3. 
    mode : either "slice" or "voxel" - defines the granularity of the normalization.
        Voxelwise normalization does not work well with only linear registered data.
    """
    print("Computing the normalization factors of the training data..")
    if mode == "slice":
        axis = (0, 1, 2, 3)
    elif mode == "voxel":
        axis = 0
    else:
        raise NotImplementedError("Normalization mode unknown.")
    samples = np.zeros(
        [len(train_idx), 1, shape[0], shape[1], shape[2]], dtype=np.float32
    )
    for c, value in enumerate(train_idx):
        samples[c] = data[value]["image"].numpy()
    mean = np.mean(samples, axis=axis)
    std = np.std(samples, axis=axis)
    return np.squeeze(mean), np.squeeze(std)


class Normalize(object):
    """
    Normalize tensor with first and second moments.
    By default will only normalize on non-zero voxels. Set 
    masked = False if this is undesired.
    """

    def __init__(self, mean, std=1, masked=True, eps=1e-10):
        self.mean = mean
        self.std = std
        self.masked = masked
        # set epsilon only if using std scaling
        self.eps = eps if np.all(std) != 1 else 0

    def __call__(self, image):
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)
        return image

    def denormalize(self, image):
        image = image * (self.std + self.eps) + self.mean
        return image

    def apply_transform(self, image):
        return (image - self.mean) / (self.std + self.eps)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero. """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.
        return image


class IntensityRescale:
    """
    Rescale image itensities between 0 and 1 for a single image.

    Arguments:
        masked: applies normalization only on non-zero voxels. Default
            is True.
        on_gpu: speed up computation by using GPU. Requires torch.Tensor
             instead of np.array. Default is False.
    """

    def __init__(self, masked=True, on_gpu=False):
        self.masked = masked
        self.on_gpu = on_gpu

    def __call__(self, image):
        if self.masked:
            image = self.zero_masked_transform(image)
        else:
            image = self.apply_transform(image)

        return image

    def apply_transform(self, image):
        if self.on_gpu:
            return normalize_float_torch(image, min=0)
        else:
            return normalize_float(image, min=0)

    def zero_masked_transform(self, image):
        """ Only apply transform where input is not zero. """
        img_mask = image == 0
        # do transform
        image = self.apply_transform(image)
        image[img_mask] = 0.
        return image


########################################################################
# Data augmentations
########################################################################


class Flip:
    """
    Flip the image along a given axis.

    Arguments:
        axis: axis to flip over. Default is 0
        prob: probability to flip the image. Executes always when set to
             1. Default is 0.5
    """
    def __init__(self, axis=0, prob=0.5):
        self.axis = axis
        self.prob = prob

    def __call__(self, image):
        rand = np.random.uniform()
        if rand <= self.prob:
            augmented = np.flip(image, axis=self.axis).copy()
        else:
            augmented = image
        return augmented


class SagittalFlip(Flip):
    """
    Flip image along the sagittal axis (x-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob=0.5):
        super().__init__(axis=0, prob=prob)
    
    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)

class CoronalFlip(Flip):
    """
    Flip image along the coronal axis (y-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob=0.5):
        super().__init__(axis=1, prob=prob)

    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)


class AxialFlip(Flip):
    """
    Flip image along the axial axis (z-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, prob=0.5):
        super().__init__(axis=2, prob=prob)

    def __call__(self, image):
        assert(len(image.shape) == 3)
        return super().__call__(image)


class Rotate:
    """ 
    Rotate the image along a given axis.

    Arguments:
        axis: axis to rotate. Default is 0
        deg: min and max rotation angles in degrees. Randomly rotates 
            within that range. Can be scalar, list or tuple. In case of 
            scalar it rotates between -abs(deg) and abs(deg). Default is
            (-3, 3).
    """
    def __init__(self, axis=0, deg=(-3, 3)):
        if isinstance(deg, tuple) or isinstance(deg, list):
            assert(len(deg) == 2)
            self.min_rot = np.min(deg)
            self.max_rot = np.max(deg)
        else:
            self.min_rot = -int(abs(deg))
            self.max_rot = int(abs(deg))

        if axis == 0:
            self.axes = (1, 0)
        elif axis == 1:
            self.axes = (2, 1)
        elif axis == 2:
            self.axes = (0, 2)

    def __call__(self, image):
        rand = np.random.randint(self.min_rot, self.max_rot + 1)
        augmented = rotate(
            image,
            angle=rand,
            axes=self.axes,
            reshape=False
            ).copy()
        return augmented


class SagittalRotate(Rotate):
    """
    Rotate image's sagittal axis (x-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=0, deg=deg)

    def __call__(self, image):
        return super().__call__(image)


class CoronalRotate(Rotate):
    """
    Rotate image's coronal axis (y-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=1, deg=deg)

    def __call__(self, image):
        return super().__call__(image)


class AxialRotate(Rotate):
    """
    Rotate image's axial axis (z-axis). 
    Expects input shape (X, Y, Z).
    """
    def __init__(self, deg=(-3, 3)):
        super().__init__(axis=2, deg=deg)

    def __call__(self, image):
        return super().__call__(image)


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
