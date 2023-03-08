import os

import pytorch_lightning as pl
from torch import optim

from utils import utils
from utils.BaseMMVae import BaseMMVae


class VAEtrimodalSVHNMNIST(BaseMMVae, pl.LightningModule):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)

    def training_step(self, batch, batch_idx):


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.flags.initial_learning_rate,
                               betas=(self.flags.beta_1, self.flags.beta_2))
        return optimizer
