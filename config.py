"""Central configuration: paths, hyper-parameters and constants.

Edit this file to adapt the project to your machine (the defaults mirror the
original Kaggle notebook layout).
"""
from pathlib import Path

# ----------------------------------------------------------------------------
# Input data (LiTS dataset, as laid out on Kaggle)
# ----------------------------------------------------------------------------
DATA_DIRS = [
    "../input/liver-tumor-segmentation",
    "../input/liver-tumor-segmentation-part-2",
]
MASK_DIR = "../input/liver-tumor-segmentation/segmentations"
NUM_VOLUMES = 131  # volume-0.nii ... volume-130.nii

# ----------------------------------------------------------------------------
# Generated data / outputs
# ----------------------------------------------------------------------------
GENERATE_JPG_FILES = True  # False -> reuse previously exported slices
OUTPUT_DIR = Path(".")  # where train_images/, train_masks/ and the model go
PRETRAINED_JPG_DIR = Path("../input/liver-segmentation-with-fastai-v2")

# Slice-export subsampling (as in the notebook)
VOLUME_STEP = 3  # use every 3rd nii file
SLICE_STEP = 2   # export every 2nd slice

# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------
BATCH_SIZE = 16
IMAGE_SIZE = 128
EPOCHS = 5
WEIGHT_DECAY = 0.1
CODES = ["background", "liver", "tumor"]  # 0, 1, 2

MODEL_NAME = "Liver_segmentation"


def data_root() -> Path:
    """Folder that contains train_images/ and train_masks/."""
    return OUTPUT_DIR if GENERATE_JPG_FILES else PRETRAINED_JPG_DIR


def train_images_dir() -> Path:
    return data_root() / "train_images"


def train_masks_dir() -> Path:
    return data_root() / "train_masks"


def model_path() -> Path:
    return data_root() / MODEL_NAME
