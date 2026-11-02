"""Build the metadata table that links each CT volume to its segmentation mask."""
import os

import pandas as pd

import config


def build_file_index() -> pd.DataFrame:
    """Walk the dataset folders and return a DataFrame with one row per CT volume.

    Columns: dirname, filename, mask_dirname, mask_filename
    """
    file_list = []
    for root in config.DATA_DIRS:
        for dirname, _, filenames in os.walk(root):
            for filename in filenames:
                file_list.append((dirname, filename))

    df_files = pd.DataFrame(file_list, columns=["dirname", "filename"])
    df_files = df_files.sort_values(by=["filename"], ascending=True)

    # Map CT scan -> mask
    df_files["mask_dirname"] = ""
    df_files["mask_filename"] = ""
    for i in range(config.NUM_VOLUMES):
        ct = f"volume-{i}.nii"
        mask = f"segmentation-{i}.nii"
        df_files.loc[df_files["filename"] == ct, "mask_filename"] = mask
        df_files.loc[df_files["filename"] == ct, "mask_dirname"] = config.MASK_DIR

    # Keep only CT volume rows (drop the segmentation rows)
    df_files = (
        df_files[df_files.mask_filename != ""]
        .sort_values(by=["filename"])
        .reset_index(drop=True)
    )
    return df_files


def ct_path(df_files: pd.DataFrame, idx: int) -> str:
    return f"{df_files.loc[idx, 'dirname']}/{df_files.loc[idx, 'filename']}"


def mask_path(df_files: pd.DataFrame, idx: int) -> str:
    return f"{df_files.loc[idx, 'mask_dirname']}/{df_files.loc[idx, 'mask_filename']}"
