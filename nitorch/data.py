import numpy as np
import nibabel as nib
from scipy.ndimage.interpolation import zoom
import matplotlib.pyplot as plt
import nibabel
from nilearn import plotting
from niwidgets import NiftiWidget
import os


def load_nifti(file_path, dtype=np.float32, incl_header=False, z_factor=None, mask=None):
    """
    Loads a volumetric image in nifti format (extensions .nii, .nii.gz etc.)
    as a 3D numpy.ndarray.
    
    Args:
        file_path: absolute path to the nifti file
        
        dtype(optional): datatype of the loaded numpy.ndarray
        
        incl_header(bool, optional): If True, the nifTI object of the image is 
        also returned
        
        z_factor(float or sequence, optional): The zoom factor along the axes. 
        If a float, zoom is the same for each axis. If a sequence, zoom should 
        contain one value for each axis.
        
        mask(ndarray, optional): A mask with the same shape as the image. 
        If provided then the mask is element-wise multiplied with the image ndarray
    
    Returns:
        3D numpy.ndarray with axis order (saggital x coronal x axial)
    """
    
    img = nib.load(file_path)
    struct_arr = img.get_data().astype(dtype)
    
    # replace infinite values with 0
    if np.inf in struct_arr:
        struct_arr[struct_arr == np.inf] = 0.
    
    # replace NaN values with 0    
    if np.isnan(struct_arr).any() == True:
        struct_arr[np.isnan(struct_arr)] = 0.
        
    if mask is not None:
        struct_arr *= mask
        
    if z_factor is not None:
        struct_arr = zoom(struct_arr, z_factor)
    
    if incl_header:
        return struct_arr, img
    else:
        return struct_arr


def show_brain(img, cut_coords=(0,0,0), 
               nifti_affine = None, interactive=False, 
               figure=None, axes=None, cmap="nipy_spectral"):
    """Displays plot cuts (by default Frontal, Axial, and Lateral) of a 3D image 
    Arg:
        img: can be (1) path to the image file stored in nifTI format
                    (2) nibabel.Nifti1Image object
                    (3) 3-dimensional numpy.ndarray
        cut_coords(optional): The MNI coordinates (in range [-90, +90]) 
        of the point where the cut will be is performed. 
        Should be a 3-tuple: (x, y, z). Default is center (0,0,0). 
        Is ignored in interactive mode.
        
        nifti_affine(optional): The 
        
        figure (optional): matplotlib figure to draw on
        axes (optional): matplotlib axes to draw on
        cmap (optional): matplotlib colormap to be used
        
        example:
            >>> f = plt.figure(figsize=(10, 4))
            >>> show_brain(img, interactive=True, figure=f)
            >>> plt.show()
        """
    
    if(isinstance(img, str) and os.path.isfile(img)) or (isinstance(img, nibabel.Nifti1Image)):
        img_nii = img
        
    elif(isinstance(img, np.ndarray)):
        assert img.ndim == 3, "The numpy.ndarray must be 3 dimensional of shape (H x W x Z)"
        
        if(nifti_affine == None):
            nifti_affine = np.eye(4)
            
        img_nii = nib.Nifti1Image(img, affine=nifti_affine)
        # convert cut co-ordinates in range [-90,90] to reflect the index range of numpy
        cut_coords = np.multiply(np.add(cut_coords, 90), tuple(img.shape))//180
    else:
        raise ValueError("Invalid value provided for 'img_pointer'- {}. Either provide a 3-dimensional numpy.ndarray of a MRI image or path to the image file stored in nifTI format.".format(type(img)))
        

    if interactive:
        widget = NiftiWidget(img_nii)
        widget.nifti_plotter(figure=figure, axes=axes, cmap=cmap)
    else:                                
        plotting.plot_img(img_nii, cut_coords, figure=figure, axes=axes, cmap=cmap)
