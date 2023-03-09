import os

from torch import optim
import pytorch_lightning as pl
from pytorch_lightning import callbacks

from utils import utils
from utils.BaseMMVae import BaseMMVae


class VAEtrimodalSVHNMNIST(BaseMMVae):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.flags.initial_learning_rate,
                               betas=(self.flags.beta_1, self.flags.beta_2))
        return optimizer


class SVHNMNISTTrainer(callbacks.Callback):
    def __init__(self, num_samples: int):
        super(SVHNMNISTTrainer, self).__init__()
        self.num_samples = num_samples

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass
