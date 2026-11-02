"""CT preprocessing: intensity windowing, histogram scaling and JPG export.

Source: https://docs.fast.ai/medical.imaging

IMPORTANT: this module monkey-patches ``torch.Tensor`` (via fastai's ``@patch``)
with ``windowed``, ``freqhist_bins``, ``hist_scaled``, ``to_nchan`` and
``save_jpg``. Import it before using those methods.
"""
import types
from pathlib import Path

import numpy as np
import torch
from fastai.vision.all import *  # noqa: F401,F403  (patch, tensor, TensorImageBW, ...)
from PIL import Image

dicom_windows = types.SimpleNamespace(
    brain=(80, 40),
    subdural=(254, 100),
    stroke=(8, 32),
    brain_bone=(2800, 600),
    brain_soft=(375, 40),
    lungs=(1500, -600),
    mediastinum=(350, 50),
    abdomen_soft=(400, 50),
    liver=(150, 30),
    spine_soft=(250, 50),
    spine_bone=(1800, 400),
    custom=(200, 60),
)

# Windows used for the 2-window (+ histogram) 3-channel input image
DEFAULT_WINDOWS = [dicom_windows.liver, dicom_windows.custom]


class TensorCTScan(TensorImageBW):
    _show_args = {"cmap": "bone"}


@patch
def windowed(self: Tensor, w, l):
    px = self.clone()
    px_min = l - w // 2
    px_max = l + w // 2
    px[px < px_min] = px_min
    px[px > px_max] = px_max
    return (px - px_min) / (px_max - px_min)


@patch
def freqhist_bins(self: Tensor, n_bins=100):
    "Split the range of pixel values into groups with ~equal pixel counts"
    imsd = self.view(-1).sort()[0]
    t = torch.cat([
        tensor([0.001]),
        torch.arange(n_bins).float() / n_bins + (1 / 2 / n_bins),
        tensor([0.999]),
    ])
    t = (len(imsd) * t).long()
    return imsd[t].unique()


@patch
def hist_scaled(self: Tensor, brks=None):
    "Scale a tensor using `freqhist_bins` to values between 0 and 1"
    if self.device.type == "cuda":
        return self.hist_scaled_pt(brks)
    if brks is None:
        brks = self.freqhist_bins()
    ys = np.linspace(0.0, 1.0, len(brks))
    x = self.numpy().flatten()
    x = np.interp(x, brks.numpy(), ys)
    return tensor(x).reshape(self.shape).clamp(0.0, 1.0)


@patch
def to_nchan(x: Tensor, wins, bins=None):
    res = [x.windowed(*win) for win in wins]
    if not isinstance(bins, int) or bins != 0:
        res.append(x.hist_scaled(bins).clamp(0, 1))
    dim = [0, 1][x.dim() == 3]
    return TensorCTScan(torch.stack(res, dim=dim))


@patch
def save_jpg(x: Tensor, path, wins, bins=None, quality=120):
    fn = Path(path).with_suffix(".jpg")
    x = (x.to_nchan(wins, bins) * 255).byte()
    im = Image.fromarray(x.permute(1, 2, 0).numpy(), mode=["RGB", "CMYK"][x.shape[0] == 4])
    im.save(fn, quality=quality)
