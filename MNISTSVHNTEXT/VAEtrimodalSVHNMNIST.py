import pytorch_lightning as pl
from utils.BaseMMVae_conf import BaseMMVae


class VAEtrimodalSVHNMNIST(BaseMMVae, pl.LightningModule):
    def __init__(self, config, modalities, subsets):
        super().__init__(config, modalities, subsets)
