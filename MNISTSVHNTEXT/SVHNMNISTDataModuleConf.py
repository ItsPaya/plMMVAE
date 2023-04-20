import os

import PIL.Image
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as f
from PIL.Image import Image as Image

from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms as transforms

from MNISTSVHNTEXT.SVHNMNISTDataset import SVHNMNIST

# BATCH_SIZE = 256


class SVHNMNISTDataModuleC(pl.LightningDataModule):
    def __init__(self, config, alphabet):
        super(SVHNMNISTDataModuleC, self).__init__()
        self.test = None
        self.config = config
        self.val = None
        self.train = None
        self.alphabet = alphabet
        self.dataset_test = None
        self.dataset_train = None
        self.dataset_val = None
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.drop_last = config.drop_last
        self.transform_mnist = transforms.Compose([transforms.ToTensor(),
                                                   transforms.ToPILImage(),
                                                   transforms.Resize(size=(28, 28), interpolation=PIL.Image.BICUBIC),
                                                   transforms.ToTensor()])
        self.transform_svhn = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage=None):
        transforms = [self.transform_mnist, self.transform_svhn]
        svhnmnist = SVHNMNIST(self.config,
                              self.alphabet,
                              train=True,
                              transform=transforms)
        print(len(svhnmnist))
        train_split = int(.8 * len(svhnmnist))
        test_split = len(svhnmnist) - train_split
        self.train, self.val = random_split(svhnmnist, [train_split, test_split])
        self.test = SVHNMNIST(self.config,
                              self.alphabet,
                              train=False,
                              transform=transforms)
        self.dataset_train = self.train
        self.dataset_val = self.val
        self.dataset_test = self.test

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)
