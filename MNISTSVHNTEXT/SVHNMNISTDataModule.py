import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as f
from PIL.Image import Image

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transforms

from MNISTSVHNTEXT.SVHNMNISTDataset import SVHNMNIST

BATCH_SIZE = 256


class SVHNMNISTDataModule(pl.LightningDataModule):
    def __init__(self, flags, alphabet):
        super(SVHNMNISTDataModule, self).__init__()
        self.flags = flags
        self.val = None
        self.train = None
        self.alphabet = alphabet
        self.dataset_test = None
        self.dataset_train = None
        self.transform_mnist = transforms.Compose([transforms.ToTensor(),
                                                   transforms.ToPILImage(),
                                                   transforms.Resize(size=(28, 28), interpolation=Image.BICUBIC),
                                                   transforms.ToTensor()])
        self.transform_svhn = transforms.Compose([transforms.ToTensor()])

    def setup(self):
        transforms = [self.transform_mnist, self.transform_svhn]
        svhnmnist = SVHNMNIST(self.flags,
                          self.alphabet,
                          train=True,
                          transform=transforms)
        self.train, self.val = random_split(svhnmnist, [55000, 5000])
        self.test = SVHNMNIST(self.flags,
                         self.alphabet,
                         train=False,
                         transform=transforms)
        self.dataset_train = self.train
        self.dataset_val = self.val
        self.dataset_test = self.test

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=BATCH_SIZE)

    def val_dataloader(self):
        return DataLoader(self.val, batch_size=BATCH_SIZE)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=BATCH_SIZE)
