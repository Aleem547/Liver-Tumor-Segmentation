"""Step 2: train the U-Net and export the model."""
import matplotlib.pyplot as plt

from dataloaders import build_dataloaders
from model import build_learner, save_model, train

if __name__ == "__main__":
    dls = build_dataloaders()
    dls.show_batch()
    plt.show()

    learn = build_learner(dls)
    train(learn)
    learn.show_results()
    plt.show()
    save_model(learn)
