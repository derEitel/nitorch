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

def normalization_factors(data, train_idx, shape):
    """ shape should be of length 3 """
    samples = np.zeros([len(train_idx), 1, shape[0], shape[1], shape[2]], dtype=np.float32)
    for idx, i in enumerate(train_idx):
        samples[idx] = (data[i]['image'].numpy())
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    return np.squeeze(mean), np.squeeze(std)

class Normalize(object):
    """Normalize tensor with first and second moments."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample, eps=1e-10):
        images, labels = sample["image"], sample["label"]
        images = (images - self.mean) / (self.std + eps)
        return {"image" : images,
                "label" : labels}
    
    def denormalize(self, sample, eps=1e-10):
        images, labels = sample["image"], sample["label"]
        images = images  * (self.std + eps) + self.mean
        return {"image" : images,
                "label" : labels}