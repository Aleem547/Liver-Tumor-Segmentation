"""Step 1: convert .nii volumes into JPG/PNG training slices."""
import config
from data_index import build_file_index
from dataset_export import export_slices

if __name__ == "__main__":
    if config.GENERATE_JPG_FILES:
        export_slices(build_file_index())
    else:
        print(f"Using existing slices from {config.PRETRAINED_JPG_DIR}")
