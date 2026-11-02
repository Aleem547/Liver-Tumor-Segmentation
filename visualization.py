"""Plotting helpers."""
import matplotlib.pyplot as plt

from ct_preprocessing import dicom_windows
from fastai.vision.all import tensor


def plot_sample(array_list, color_map="nipy_spectral"):
    """Plot a CT slice (array_list[0]) with its mask (array_list[1])."""
    plt.figure(figsize=(20, 16), dpi=100)

    plt.subplot(1, 4, 1)
    plt.imshow(array_list[0], cmap="bone")
    plt.title("Original Image")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(
        tensor(array_list[0].astype("float32")).windowed(*dicom_windows.liver),
        cmap="bone",
    )
    plt.title("Windowed Image")
    plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title("Mask")
    plt.axis("off")

    plt.subplot(1, 4, 4)
    plt.imshow(array_list[0], cmap="bone")
    plt.imshow(array_list[1], alpha=0.5, cmap=color_map)
    plt.title("Liver & Mask")
    plt.axis("off")

    plt.show()
