"""Reading NIfTI (.nii) CT volumes and masks."""
import nibabel as nib
import numpy as np


def read_nii(filepath):
    """Read a .nii file and return the (rotated) pixel array."""
    ct_scan = nib.load(filepath)
    array = ct_scan.get_fdata()
    array = np.rot90(np.array(array))
    return array
