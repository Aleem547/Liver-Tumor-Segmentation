"""U-Net (ResNet-50 encoder) learner construction, training and export."""
from fastai.vision.all import *  # noqa: F401,F403

import config
from metrics import cust_foreground_acc, foreground_acc


def build_learner(dls):
    return unet_learner(
        dls,
        resnet50,
        loss_func=CrossEntropyLossFlat(axis=1),
        metrics=[foreground_acc, cust_foreground_acc],
    )


def train(learn):
    learn.fine_tune(config.EPOCHS, wd=config.WEIGHT_DECAY, cbs=SaveModelCallback())
    return learn


def save_model(learn):
    learn.export(config.model_path())
