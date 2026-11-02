"""fastai DataBlock / DataLoaders for the exported slices."""
from fastai.vision.all import *  # noqa: F401,F403

import config


def get_x(fname: Path):
    return fname


def label_func(x):
    """Map an image path to its mask path."""
    return config.train_masks_dir() / f"{x.stem}_mask.png"


def build_datablock():
    codes = np.array(config.CODES)  # {background: 0, liver: 1, tumor: 2}
    tfms = [IntToFloatTensor(), Normalize()]
    return DataBlock(
        blocks=(ImageBlock(), MaskBlock(codes)),
        batch_tfms=tfms,
        splitter=RandomSplitter(),
        item_tfms=[Resize(config.IMAGE_SIZE)],
        get_items=get_image_files,
        get_y=label_func,
    )


def build_dataloaders():
    db = build_datablock()
    return db.dataloaders(config.train_images_dir(), bs=config.BATCH_SIZE)  # num_workers=0 if needed
