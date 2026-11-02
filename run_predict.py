"""Step 3: test the model on a full CT volume and plot one predicted slice."""
import sys

import matplotlib.pyplot as plt

from data_index import build_file_index, ct_path, mask_path
from inference import load_model, predict_volume
from nifti_io import read_nii
from visualization import plot_sample

if __name__ == "__main__":
    tst = int(sys.argv[1]) if len(sys.argv) > 1 else 20          # volume index
    slice_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 500   # slice index

    df_files = build_file_index()
    learn = load_model()

    test_nii = read_nii(ct_path(df_files, tst))
    test_mask = read_nii(mask_path(df_files, tst))
    print(test_nii.shape)
    plot_sample([test_nii[..., slice_idx], test_mask[..., slice_idx]])

    predicted_mask = predict_volume(learn, ct_path(df_files, tst))
    print("Number of test slices:", len(predicted_mask))
    plt.imshow(predicted_mask[slice_idx])
    plt.title("Predicted mask")
    plt.show()
