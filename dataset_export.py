"""Export CT volumes to 2D JPG slices + PNG masks for U-Net training."""
import os

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

import config
from ct_preprocessing import DEFAULT_WINDOWS  # also registers Tensor patches
from data_index import ct_path, mask_path
from fastai.vision.all import tensor
from nifti_io import read_nii


def export_slices(df_files):
    """Write train_images/*.jpg and train_masks/*_mask.png.

    The full dataset has 131 nii files / 67072 slices; by default only every
    VOLUME_STEP-th volume and every SLICE_STEP-th slice is exported.
    """
    img_dir = config.train_images_dir()
    mask_dir = config.train_masks_dir()
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    for ii in tqdm(range(0, len(df_files), config.VOLUME_STEP)):
        curr_ct = read_nii(ct_path(df_files, ii))
        curr_mask = read_nii(mask_path(df_files, ii))
        curr_file_name = str(df_files.loc[ii, "filename"]).split(".")[0]
        curr_dim = curr_ct.shape[2]  # 512, 512, curr_dim

        for curr_slice in range(0, curr_dim, config.SLICE_STEP):
            data = tensor(curr_ct[..., curr_slice].astype(np.float32))
            mask = Image.fromarray(curr_mask[..., curr_slice].astype("uint8"), mode="L")
            data.save_jpg(img_dir / f"{curr_file_name}_slice_{curr_slice}.jpg", DEFAULT_WINDOWS)
            mask.save(mask_dir / f"{curr_file_name}_slice_{curr_slice}_mask.png")
