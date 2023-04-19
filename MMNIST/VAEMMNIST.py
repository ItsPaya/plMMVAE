from utils.BaseMMVae_conf import BaseMMVae
import pytorch_lightning as pl


class VAEMMNIST(BaseMMVae, pl.LightningModule):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)
