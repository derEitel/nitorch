import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom


def load_nifti(file_path, z_factor=None, dtype=None, incl_header=False, mask=None):
    if dtype is None:
        dt = np.float32
    else:
        dt = dtype
    img = nib.load(file_path)
    struct_arr = img.get_data().astype(dt)
    if mask is not None:
        struct_arr *= mask
    if z_factor is not None:
        struct_arr = zoom(struct_arr, z_factor)
    if incl_header:
        return struct_arr, img
    else:
        return struct_arr


def normalize_float(x, min=-1):
    """ Function to normalize a matrix of floats. """
    if min == -1:
        norm = 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1
        return norm
    elif min == 0:
        norm = (x - np.min(x)) / (np.max(x) - np.min(x))
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

    def __call__(self, sample):
        images, labels = sample["image"], sample["label"]
        if self.masked:
            images = self.zero_masked_transform(images)
        else:
            images = self.apply_transform(images)
        return {"image": images, "label": labels}

    def denormalize(self, sample):
        images, labels = sample["image"], sample["label"]
        images = images * (self.std + self.eps) + self.mean
        return {"image": images, "label": labels}

    def apply_transform(self, images):
        return (images - self.mean) / (self.std + self.eps)

    def zero_masked_transform(self, images):
        """ Only apply transform where input is not zero. """
        img_mask = images == 0
        # do transform
        images = self.apply_transform(images)
        images[img_mask] = 0.
        return images


class IntensityRescale:
    """
    By default will only normalize on non-zero voxels. Set 
    masked = False if this is undesired.
    """

    def __init__(self, masked=True):
        self.masked = masked

    def __call__(self, batch):
        images, labels = batch["image"], batch["label"]
        if self.masked:
            images = self.zero_masked_transform(images)
        else:
            images = self.apply_transform(images)

        return {"image": images, "label": labels}

    def apply_transform(self, images):
        return normalize_float(images, min=0)

    def zero_masked_transform(self, images):
        """ Only apply transform where input is not zero. """
        img_mask = images == 0
        # do transform
        images = self.apply_transform(images)
        images[img_mask] = 0.
        return images
