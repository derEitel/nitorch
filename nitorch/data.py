import numpy as np
import nibabel as nib
import nilearn.image as nilimg
from scipy.ndimage.interpolation import zoom
import nibabel
import os
import warnings
import logging
import multiprocessing
from pathlib import Path
from tabulate import tabulate
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from tqdm import tqdm_notebook as tqdm
import pickle
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose
import matplotlib.pyplot as plt
from nitorch.transforms import *

logger = logging.getLogger(__name__)

def balanced_subsample(y, size=None):
    subsample = []
    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()
    return subsample


def load_nifti(file_path, dtype=np.float32, incl_header=False, z_factor=None, mask=None, force_to_shape=None, remove_nan=True):
    """
    Loads a volumetric image in nifti format (extensions .nii, .nii.gz etc.)
    as a 3D numpy.ndarray.
    
    Args:
        file_path: absolute path to the nifti file
        
        dtype(optional): datatype of the loaded numpy.ndarray
        
        incl_header(bool, optional): If True, the nifTI object of the 
        image is also returned.
        
        z_factor(float or sequence, optional): The zoom factor along the
        axes. If a float, zoom is the same for each axis. If a sequence,
        zoom should contain one value for each axis.
        
        mask(ndarray, optional): A mask with the same shape as the
        original image. If provided then the mask is element-wise
        multiplied with the image ndarray

        force_to_shape(list, optional): 3-dimensional desired shape of the
        nifti. if specified, nifti will be forced to have these dimension by
        cutting the edges of the images, evenly on both sides of each dimension

        remove_nan(bool, optional): If True, NaN values will be converted
        to num
    
    Returns:
        3D numpy.ndarray with axis order (saggital x coronal x axial)
    """
    
    img = nib.load(file_path)

    if force_to_shape:
        img = _force_to_shape(img, force_to_shape)

    struct_arr = img.get_data().astype(dtype)
    
    # replace infinite values with 0
    if np.inf in struct_arr:
        struct_arr[struct_arr == np.inf] = 0.

    if remove_nan:
        struct_arr = np.nan_to_num(struct_arr)
    
    # replace NaN values with 0    
    if np.isnan(struct_arr).any() == True:
        struct_arr[np.isnan(struct_arr)] = 0.
        
    if z_factor is not None:
        if isinstance(z_factor, float):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                struct_arr = np.around(zoom(struct_arr, z_factor), 0)
        else:
            TypeError("z_factor has to be one of None or tuple")
    if mask is not None: struct_arr *= mask
    
    if incl_header:
        return struct_arr, img
    else:
        return struct_arr


def _force_to_shape(img_nii_X, shape, rtol=1e-8, copy=True):
    """
    forces image to have a certain shape. Retuns cutted image.

    Args:
        img_nii_X: Niimg-like object

        shape(list): desired shape of the nifti. 3-Dimensions -> 3
        entries in the list

        rtol(float, optional): dont keep values under these threshold.
        they only make calculation unnecessary difficult

        copy(bool, optional): Specifies whether cropped data is to
        be copied or not.

    Returns:
        3D numpy.ndarray with axis order (saggital x coronal x axial)
    """
    data = img_nii_X.get_data()
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    # add as many voxel as neccessary to archive shape
    missing_diff_start = []
    missing_diff_end = []
    if shape:
        assert shape < list(data.shape), "Shape of the data smaller than shape it should be transformed to!"
        diff = end - start
        missing_diff = shape - diff
        for dim in missing_diff:
            if dim % 2 == 0:
                missing_diff_start.append(int(dim / 2))
                missing_diff_end.append(int(dim / 2))
            else:
                missing_diff_start.append(int(dim / 2))  # conversion to int always floors the value
                # to archive desired shape we have to add what we floored before
                missing_diff_end.append(int(dim / 2) + 1)

        for idx, s in enumerate(start):
            if (s - missing_diff_start[idx]) > 0:
                start[idx] = s - missing_diff_start[idx]
            else:
                start[idx] = 0
                # add rest of the pixels to end
                diff = missing_diff_start[idx] - s
                end[idx] = end[idx] + diff
        end = end + missing_diff_end

    slices = [slice(s, e) for s, e in zip(start, end)]

    return _crop_img_to(img_nii_X, slices, copy=copy)


def _crop_img_to(img, slices, copy=True):
        """Crops image to a smaller size

        Crop img to size indicated by slices and adjust affine
        accordingly

        Parameters
        ----------
        img: Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Img to be cropped. If slices has less entries than img
            has dimensions, the slices will be applied to the first len(slices)
            dimensions

        slices: list of slices
            Defines the range of the crop.
            E.g. [slice(20, 200), slice(40, 150), slice(0, 100)]
            defines a 3D cube

        copy: boolean
            Specifies whether cropped data is to be copied or not.
            Default: True

        Returns
        -------
        cropped_img: Niimg-like object
            See http://nilearn.github.io/manipulating_images/input_output.html
            Cropped version of the input image
        """

        data = img.get_data()
        affine = img.affine

        cropped_data = data[tuple(slices)]
        if copy:
            cropped_data = cropped_data.copy()

        linear_part = affine[:3, :3]
        old_origin = affine[:3, 3]
        new_origin_voxel = np.array([s.start for s in slices])
        new_origin = old_origin + linear_part.dot(new_origin_voxel)

        new_affine = np.eye(4)
        new_affine[:3, :3] = linear_part
        new_affine[:3, 3] = new_origin

        return nilimg.new_img_like(img, cropped_data, new_affine)


class MRIDataset(Dataset):
    """
    PyTorch dataset that consists of MRI images and labels.

    Args:
        filenames (iterable of strings): The filenames to the MRI images.
        labels (iterable): The labels for the images.
        mask (array): If not None (default), images are masked by multiplying with this array.
        transform: Any transformations to apply to the images.
    """

    def __init__(self, filenames, labels, id2label, z_factor=None, mask=None, transform=None, force_to_shape=None):
        self.filenames = filenames
        self.labels = torch.FloatTensor(labels)
        self.label_counts = dict(zip(*np.unique(labels, return_counts=True)))
        self.class_weights = np.array(list(self.label_counts.values())) / len(labels)

        self.mask = mask

        self.force_to_shape = force_to_shape

        self.transform = transform

        self.id2label = id2label
        self.z_factor = z_factor
        # Required by torchsample.
        self.num_inputs = 1
        self.num_targets = 1

        # Default values. Should be set via fit_normalization.
        self.mean = 0
        self.std = 1

        self.shape = self.get_image_shape()

    def __len__(self):
        return len(self.filenames)

    def __repr__(self):

        return (f"MRIDataset - no. samples: {len(self)}; shape: {self.shape}; no. classes: {len(self.labels.unique())}")

    def __getitem__(self, idx):
        """Return the image as a FloatTensor and its corresponding label."""
        label = self.labels[idx].unsqueeze_(-1)  # add the missing dimension

        struct_arr = load_nifti(
            self.filenames[idx], mask=self.mask, z_factor=self.z_factor, dtype=np.float32,
            force_to_shape=self.force_to_shape)
        # TDOO: Try normalizing each image to mean 0 and std 1 here.
        # struct_arr = (struct_arr - struct_arr.mean()) / (struct_arr.std() + 1e-10)
        # prevent 0 division by adding small factor

        if self.transform is not None:
            struct_arr = self.transform(struct_arr)
        else:
            struct_arr = (struct_arr - self.mean) / (self.std + 1e-10)
            struct_arr = torch.FloatTensor(struct_arr[None])  # add (empty) channel dimension

        return struct_arr, label

    def get_image_shape(self):
        """The shape of the MRI images."""
        img = load_nifti(self.filenames[0], mask=None, z_factor=self.z_factor, force_to_shape=self.force_to_shape)
        return img.shape

    def fit_normalization(self, num_sample=None, show_progress=False):
        """
        Calculate the voxel-wise mean and std across the dataset for normalization.

        Args:
            num_sample (int or None): If None (default), calculate the values across the complete dataset,
                                      otherwise sample a number of images.
            show_progress (bool): Show a progress bar during the calculation."
        """
        if num_sample is None:
            num_sample = len(self)
        image_shape = self.get_image_shape()
        all_struct_arr = np.zeros(
            (num_sample, image_shape[0], image_shape[1], image_shape[2]))
        sampled_filenames = np.random.choice(
            self.filenames, num_sample, replace=False)

        if show_progress:
            sampled_filenames = tqdm(sampled_filenames)

        for i, filename in enumerate(sampled_filenames):
            all_struct_arr[i] = load_nifti(filename, mask=self.mask, z_factor=self.z_factor,
                                           force_to_shape=self.force_to_shape)

        self.mean = all_struct_arr.mean(0)
        self.std = all_struct_arr.std(0)

    def get_raw_image(self, idx):
        """Return the raw image at index idx (i.e. not normalized, no color channel, no transform."""
        return load_nifti(self.filenames[idx], mask=self.mask, z_factor=self.z_factor,
                          force_to_shape=self.force_to_shape)


def get_image_filepath(df_row, source_dir=''):
    """Return the filepath of the image that is described in the row of the data table."""
    # Current format for the image filepath is:
    # <PTID>/<Visit (spaces removed)>/<PTID>_<Scan.Date (/ replaced by -)>_<Visit (spaces removed)>_<Image.ID>_<DX>_Warped.nii.gz
    filedir = os.path.join(df_row['PTID'], df_row['Visit'].replace(' ', ''))
    filename = '{}_{}_{}_{}_{}_Warped.nii.gz'.format(df_row['PTID'], df_row['Scan.Date'].replace(
        '/', '-'), df_row['Visit'].replace(' ', ''), df_row['Image.ID'], df_row['DX'])
    return os.path.join(source_dir, filedir, filename)


class DataBunch():
    DEFAULT_FILE = 'file_path'
    DEFAULT_LABEL = 'DX'
    DEFAULT_PTID = 'PTID'

    CACHE_NAME = 'databunch.pkl'

    def __init__(self, source_dir: str, path: str, table: str, image_dir: str = None, mask: str = None,
                 transforms: Compose = Compose([ToTensor(), IntensityRescale(masked=False, on_gpu=True)]),
                 labels_to_keep: list = None, get_file_path: callable = None, balance: bool = False,
                 num_samples: int = None,
                 num_training_samples: int = None, z_factor: float = 0.5, test_size: float = 0.1, grouped: bool = False,
                 no_cache: bool = True, force_to_shape: list = None, file_col='file_path', label_col='DX',
                 ptid_col='PTID', random_state: int = 42, **kwargs):

        """DataBunch class to built training and test MRIDatasets and DataLoaders from a single input csv file containing .nii file paths.
        Upon initialization, test set is randomly picked based on arguments grouped,balanced and test_size.
        Important methods:
            - normalize: normalize dataset based on training data.
            - build_dataloaders: re-batchify data and store iterators at `train_dl` and `test_dl`.
            - print_stats: prints set and patient level statistics
            - show_sample: show random processed training sample

        # Arguments:
            source_dir: Path to source_dir folder, where table and image_dir can be found.
            path: Path where intermediary data will be stored (eg. cache).
            image_dir: Image directory *relative* to source_dir, where the .nii files are.
            table: CSV file path *relative* to source_dir containing samples. The tables *must*
                   contain file_col, label_col and ptid_col columns.
            mask: Path to binary brain mask in .nii format. This will be resized with z_factor.
            transforms: A PyTorch Compose container object, with the transformations to apply to samples. Defaults to
                        using ToTensor() and IntensityRescaling() into [0,1].
            labels_to_keep: List of labels to keep in the datasets. Defaults to None (all labels).
            get_file_path: A function mapping the rows of table to the respective file paths of the samples.
            balance: Boolean switch for enforcing balanced classes.
            grouped: Boolean switch to enforce grouped train/test splitting, i.e. ensuring that no train samples
                     are present in the test set.
            test_size: Fraction of samples to pick for test set.
            num_samples: Total no. of samples to consider from the table., defaults to None (all).
            num_training_samples: No. of training samples to pick, defaults to None (all).
            z_factor: Zoom factor to apply to each image.
            no_cache: Prevents caching (caching is useful when later we normalize the DataBunch and load it back).
            file_col: Column name in table identifying the path to the given sample's .nii file.
            label_col: Column name in table identifying the path to the given sample's label.
            ptid_col: Column name in table identifying the path to the given sample's patient ID.
            random_state: Random state to enforce reproducibility for train/test splitting.


        """

        self.set_column_ids(file_col, label_col, ptid_col)

        if not os.path.isdir(source_dir):
            raise RuntimeError(f"{source_dir} not existing!")
        self.source_dir = Path(source_dir)
        self.path = Path(path)
        if not no_cache:
            if os.path.exists(self.path / self.CACHE_NAME):
                ans = str(input(f"Do you want to load cache from {self.path / self.CACHE_NAME}? [y/n]")).strip()
                if ans == 'y':
                    try:
                        self.load()
                        self.loaded_cache = True
                        self.print_stats()
                        print(f"DataBunch initialized at {self.path}")
                        return
                    except EOFError:
                        logger.warning("Pickled DataBunch is corrupted at {}".format(self.path))
                        print(f"Cannot load {self.CACHE_NAME} because it is corrupted. Building Databunch..\n")

                elif ans == 'n':
                    pass
                else:
                    raise RuntimeError(f"Invalid answer {ans}.")
        self.loaded_cache = False
        os.makedirs(path, exist_ok=True)
        self.table = table
        self.image_dir = self.source_dir / image_dir if image_dir is not None else None
        self.z_factor = z_factor

        self.force_to_shape = force_to_shape

        self.mask = load_nifti(
            str(mask), z_factor=z_factor, force_to_shape=self.force_to_shape) if mask is not None else None

        self.random_state = random_state
        df = pd.read_csv(self.source_dir / self.table, index_col=None)
        print(f"Found {len(df)} images in {self.table}")
        print(
            f"Found {len(df[self.LABEL].unique())} labels: {df[self.LABEL].unique().tolist()}")

        if balance:
            subsample_idx = balanced_subsample(df[self.LABEL])
            df = df[df.index.isin(subsample_idx)]

        get_file_path = get_image_filepath if get_file_path is None else get_file_path
        if self.FILE not in df.columns:
            if get_file_path is not None and self.image_dir is not None and callable(get_file_path):
                df[self.FILE] = df.apply(
                    lambda r: get_file_path(r, self.image_dir), axis=1)
            else:
                raise RuntimeError(f"If {self.FILE} column is not in {self.table},"
                                   f"please pass a valid `get_file_path` function and an `image_dir`.")
        len_before = len(df)
        self.labels_to_keep = df[self.LABEL].unique().tolist() if labels_to_keep is None else labels_to_keep
        df = df[df[self.LABEL].isin(self.labels_to_keep)]

        print(
            f"Dropped {len_before - len(df)} samples that were not in {self.labels_to_keep}")
        self.df = df[[self.FILE, self.LABEL, self.PTID]].dropna()

        print(
            f"Final dataframe contains {len(self.df)} samples from {len(df[self.PTID].unique())} patients")
        self.classes = self.df[self.LABEL].unique().tolist()[::-1]
        self.label2id = {k: v for k, v in zip(
            self.classes, np.arange(len(self.classes)))}
        self.id2label = dict(zip(self.label2id.values(), self.label2id.keys()))
        self.test_size = test_size
        self.transforms = transforms
        if test_size is not None:
            self.build_datasets(test_size=test_size, transforms=transforms, num_samples=num_samples,
                                num_training_samples=num_training_samples, grouped=grouped)
            self.print_stats()
        print(f"DataBunch initialized at {self.path}")

    def set_column_ids(self, file_col, label_col, ptid_col):
        self.FILE = self.DEFAULT_FILE if file_col is None else file_col
        self.LABEL = self.DEFAULT_LABEL if label_col is None else label_col
        self.PTID = self.DEFAULT_PTID if ptid_col is None else ptid_col
        logger.info(f"Using file column {self.FILE}; label column {self.LABEL} and patient_id column {self.PTID}")

    def build_datasets(self, test_size: float = .1, transforms: list = None, num_samples=None,
                       num_training_samples=None, random_state: int = None, grouped=False):
        print("Building datasets")
        print(
            f"Patient-wise train/test splitting with test_size = {test_size}")
        random_state = self.random_state if random_state is None else random_state

        if num_samples is not None:
            self.df = self.df.sample(n=num_samples)
            logger.info(f"Sampling {num_training_samples} samples")

        if grouped:
            gss = GroupShuffleSplit(
                n_splits=1, test_size=test_size, random_state=random_state)
            trn, tst = next(
                iter(gss.split(self.df, groups=self.df[self.PTID].tolist())))
            df_trn, df_tst = self.df.iloc[trn, :], self.df.iloc[tst, :]

        else:
            df_trn, df_tst = train_test_split(self.df, test_size=test_size, stratify=self.df[self.LABEL], shuffle=True)

        self.df_trn, self.df_tst = df_trn, df_tst

        if num_training_samples is not None:
            self.df_trn = self.df_trn.sample(n=num_training_samples)
            logger.info(f"Sampling {num_training_samples} training samples")

        self.train_ds = MRIDataset(self.df_trn[self.FILE].tolist(),
                                   [self.label2id[l]
                                    for l in df_trn[self.LABEL]],
                                   id2label=self.id2label,
                                   z_factor=self.z_factor,
                                   transform=transforms,
                                   force_to_shape=self.force_to_shape,
                                   mask=self.mask)
        self.test_ds = MRIDataset(df_tst[self.FILE].tolist(),
                                  [self.label2id[l]
                                   for l in df_tst[self.LABEL]],
                                  id2label=self.id2label,
                                  z_factor=self.z_factor,
                                  transform=transforms,
                                  force_to_shape=self.force_to_shape,
                                  mask=self.mask)

        self.shape = self.train_ds.shape

        self.train_dl, self.test_dl = None, None

    def normalize(self, use_samples: int = None):
        """Normalizes the dataset with mean and std calculated on the training set"""

        if not hasattr(self, "train_ds"):
            raise RuntimeError(f"Attribute `train_ds` not found.")
        print("Normalizing datasets")

        if use_samples is None:
            use_samples = len(self.train_ds)
        else:
            use_samples = len(self.train_ds) if use_samples > len(
                self.train_ds) else use_samples
        print(
            f"Calculating mean and std for normalization based on {use_samples} train samples:")
        self.train_ds.fit_normalization(
            num_sample=use_samples, show_progress=True)
        self.test_ds.mean, self.test_ds.std = self.train_ds.mean, self.train_ds.std
        self.mean, self.std = self.train_ds.mean, self.train_ds.std
        self.test_ds.mean, self.test_ds.std = self.mean, self.std
        self.train_ds.transform = None
        self.test_ds.transform = None

    def build_dataloaders(self, bs: int = 8, normalize: bool = False, use_samples: int = None, num_workers: int = None):
        """Build DataLoaders with bs, optionally normalizing the datasets too, or performing downsampling."""

        print("Building dataloaders")

        if normalize:
            if self.loaded_cache:
                print("Already normalized -- using attributes `mean` and `std`.")
            else:
                self.normalize(use_samples=use_samples)

        else:
            logger.warning("Dataset not normalized, performance might be significantly hurt!")
        print(
            f"No. training/test samples: {len(self.train_ds)}/{len(self.test_ds)}")

        if num_workers is None: num_workers = multiprocessing.cpu_count()
        pin_memory = torch.cuda.is_available()
        self.train_dl = DataLoader(self.train_ds, batch_size=bs, shuffle=True,
                                   num_workers=num_workers, pin_memory=pin_memory)
        self.test_dl = DataLoader(self.test_ds, batch_size=bs, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)

    def print_stats(self):
        """Print statistics about the patients and images."""
        headers = []
        headers.append('IMAGES')
        headers += [cls for cls in self.classes]
        headers.append('PATIENTS')
        headers += [cls for cls in self.classes]

        def get_stats(df):
            image_count, patient_count = [
                                             len(df)], [len(df[self.PTID].unique())]
            image_count += [len(df[df[self.LABEL] == cls])
                            for cls in self.classes]
            patient_count += [len(df[df[self.LABEL] == cls]
                                  [self.PTID].unique()) for cls in self.classes]
            return image_count + patient_count

        stats = [['Train'] + get_stats(self.df_trn),
                 ['Test'] + get_stats(self.df_tst),
                 ['Total'] + get_stats(self.df)]
        print(tabulate(stats, headers=headers))
        print()
        print(f"Data shape: {self.train_ds.shape}")
        if self.z_factor is not None:
            print(f"NOTE: data have been downsized by a factor of {self.z_factor}")

    def show_sample(self, cmap="gray", **kwargs):
        """Shows a random training sample after zooming, masking and tranformations."""

        if self.train_ds is None:
            raise RuntimeError(
                f"`train_ds` not found, please call `build` method first.")
        img, lbl = self.train_ds[np.random.randint(0, len(self.train_ds))]
        print(f"label={self.id2label[lbl.item()]}")
        f = show_brain(img[0].numpy(), cmap=cmap)
        plt.show()

    def save(self):
        """Cache the entire DataBunch object to `path`."""

        with open(self.path / self.CACHE_NAME, 'wb', closefd=True) as file:
            pickle.dump(self.__dict__,
                        file,
                        protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Saved DataBunch to {self.path / self.CACHE_NAME}")

    def load(self):
        """Load cached DataBunch object from `path`."""

        with open(self.path / self.CACHE_NAME, 'rb') as file:
            tmp_dict = pickle.load(file)
            self.__dict__.update(tmp_dict)
        print(f"Cached DataBunch has been successfully loaded.")


def show_brain(img, cut_coords=None,
               figsize=(10,5), cmap="nipy_spectral",
               draw_cross = True,
               return_fig = False
               ):
    """Displays 2D cross-sections of a 3D image along all 3 axis
    Arg:
        img: can be (1) 3-dimensional numpy.ndarray
                    (2) nibabel.Nifti1Image object
                    (3) path to the image file stored in nifTI format

        cut_coords (optional): The voxel coordinates
        of the axes where the cross-section cuts will be performed. 
        Should be a 3-tuple: (x, y, z). Default is the center = img_shape/2 
        
        figsize (optional): matplotlib figsize. Default is (10,5)
        cmap (optional): matplotlib colormap to be used
        
        draw_cross (optional): Draws horizontal and vertical lines which
        show where the cross-sections have been performed. D
        

        example:
            >>> show_brain(img, figsize=(7, 3), draw_cross=False)
            >>> plt.show()
        """
    
    if(isinstance(img, str) and os.path.isfile(img)):
        img_arr = load_nifti(img)
    elif(isinstance(img, nibabel.Nifti1Image)):
        img_arr = img.get_data()
        
    elif(isinstance(img, np.ndarray)):
        assert img.ndim == 3, "The numpy.ndarray must be 3-dimensional with shape (H x W x Z)"
        img_arr = img
    else:
        raise TypeError("Invalid type provided for 'img'- {}. \
Either provide a 3-dimensional numpy.ndarray of a MRI image or path to \
the image file stored as a nifTI format.".format(type(img)))
        
    # print(img_arr.shape)
    # img_arr = np.moveaxis(img_arr, 0, 1)
    # print(img_arr.shape)

    x_len, y_len, z_len = img_arr.shape
    # if cut_coordinates is not specified set it to the center of the image
    if(cut_coords == None):
        cut_coords = (x_len//2, y_len//2, z_len//2)

    f, ax = plt.subplots(nrows=1, ncols=3, figsize=figsize)

    ax[0].set_title("Saggital cross-section at x={}".format(cut_coords[0]))
    ax[0].imshow(
         np.rot90(img_arr[cut_coords[0],:,:]), cmap=cmap, aspect="equal")
    #draw cross
    if(draw_cross):
        ax[0].axvline(x=cut_coords[1], color='k', linewidth=1)
        ax[0].axhline(y=cut_coords[2], color='k', linewidth=1)

    ax[1].set_title("Coronal cross-section at y={}".format(cut_coords[1]))
    ax[1].imshow(
        np.rot90(img_arr[:,cut_coords[1],:]), cmap=cmap, aspect="equal")
    ax[1].text(0.05, 0.95,'L', 
        horizontalalignment='left', verticalalignment='top',
        transform=ax[1].transAxes
        , bbox=dict(facecolor='white')
        )
    ax[1].text(0.95, 0.95,'R', 
        horizontalalignment='right', verticalalignment='top'
        , transform=ax[1].transAxes
        , bbox=dict(facecolor='white')
        )
    #draw cross
    if(draw_cross):
        ax[1].axvline(x=cut_coords[0], color='k', linewidth=1)
        ax[1].axhline(y=cut_coords[2], color='k', linewidth=1)

    ax[2].set_title("Axial cross-section at z={}".format(cut_coords[2]))
    ax[2].imshow(
        np.rot90(img_arr[:,:,cut_coords[2]]), cmap=cmap, aspect="equal"
        )
    ax[2].text(0.05, 0.95,'L'
        , horizontalalignment='left', verticalalignment='top'
        , transform=ax[2].transAxes
        , bbox=dict(facecolor='white')
        )
    ax[2].text(0.95, 0.95,'R', 
        horizontalalignment='right', verticalalignment='top'
        , transform=ax[2].transAxes
        , bbox=dict(facecolor='white')
        )
    #draw cross
    if(draw_cross):
        ax[2].axvline(x=cut_coords[0], color='k', linewidth=1)
        ax[2].axhline(y=cut_coords[1], color='k', linewidth=1)

    plt.tight_layout()
    if return_fig:
        return f
