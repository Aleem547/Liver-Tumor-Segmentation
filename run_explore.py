"""Data exploration (notebook cells 7-13, 17-19): inspect a sample volume."""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import config
from ct_preprocessing import dicom_windows
from data_index import build_file_index, ct_path, mask_path
from fastai.vision.all import tensor
from nifti_io import read_nii
from visualization import plot_sample

if __name__ == "__main__":
    df_files = build_file_index()
    print(df_files)

    sample_ct = read_nii(ct_path(df_files, 0))
    sample_mask = read_nii(mask_path(df_files, 0))
    print(f"CT Shape:   {sample_ct.shape}\nMask Shape: {sample_mask.shape}")
    print(np.amin(sample_ct), np.amax(sample_ct))
    print(np.amin(sample_mask), np.amax(sample_mask))

    # Windowed view of one slice
    plt.figure(figsize=(8, 6), dpi=100)
    plt.imshow(tensor(sample_ct[..., 55].astype(np.float32)).windowed(*dicom_windows.liver), cmap="bone")
    plt.show()

    plot_sample([sample_ct[..., 55], sample_mask[..., 55]])

    # Mask value distribution
    mask = Image.fromarray(sample_mask[..., 55].astype("uint8"), mode="L")
    unique, counts = np.unique(mask, return_counts=True)
    print(np.array((unique, counts)).T)
