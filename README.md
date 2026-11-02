# Liver & Tumor Segmentation (U-Net / ResNet-50, fastai)

## Layout
| File | Purpose |
|---|---|
| `config.py` | Paths, hyper-parameters, constants |
| `data_index.py` | Builds the CT-volume ↔ mask table |
| `nifti_io.py` | `.nii` reader |
| `ct_preprocessing.py` | HU windowing, histogram scaling, JPG export (patches `Tensor`) |
| `visualization.py` | `plot_sample` |
| `dataset_export.py` | Exports 2D slices (jpg) + masks (png) |
| `dataloaders.py` | fastai `DataBlock` / `DataLoaders` |
| `metrics.py` | `foreground_acc`, `cust_foreground_acc` |
| `model.py` | U-Net learner, training, export |
| `inference.py` | Load model, predict a whole volume |
| `run_explore.py` | Optional: inspect a sample |
| `run_export_slices.py` | Step 1 |
| `run_train.py` | Step 2 |
| `run_predict.py` | Step 3 (`python run_predict.py [volume_idx] [slice_idx]`) |

## Usage
```bash
pip install -r requirements.txt
python run_export_slices.py
python run_train.py
python run_predict.py 20 500
```
Run all commands from the project root.
