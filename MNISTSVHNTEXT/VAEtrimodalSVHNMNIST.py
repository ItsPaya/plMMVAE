import os

from torch import optim

from utils import utils
from utils.BaseMMVae import BaseMMVae


class VAEtrimodalSVHNMNIST(BaseMMVae):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)
