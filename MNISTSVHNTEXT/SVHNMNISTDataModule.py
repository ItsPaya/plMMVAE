import PIL.Image

import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms

from MNISTSVHNTEXT.SVHNMNISTDataset import SVHNMNIST


class SVHNMNISTDataModule(pl.LightningDataModule):
    def __init__(self, flags, alphabet):
        super(SVHNMNISTDataModule, self).__init__()
        self.test = None
        self.flags = flags
        self.val = None
        self.train = None
        self.alphabet = alphabet
        self.dataset_test = None
        self.dataset_train = None
        self.dataset_val = None
        self.transform_mnist = transforms.Compose([transforms.ToTensor(),
                                                   transforms.ToPILImage(),
                                                   transforms.Resize(size=(28, 28), interpolation=PIL.Image.BICUBIC),
                                                   transforms.ToTensor()])
        self.transform_svhn = transforms.Compose([transforms.ToTensor()])

    def setup(self, stage=None):
        transforms = [self.transform_mnist, self.transform_svhn]
        svhnmnist = SVHNMNIST(self.flags,
                              self.alphabet,
                              train=True,
                              transform=transforms)
        print(len(svhnmnist))
        train_split = int(.8 * len(svhnmnist))
        test_split = len(svhnmnist) - train_split
        self.train, self.val = random_split(svhnmnist, [train_split, test_split])
        self.test = SVHNMNIST(self.flags,
                              self.alphabet,
                              train=False,
                              transform=transforms)
        self.dataset_train = self.train
        self.dataset_val = self.val
        self.dataset_test = self.test

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.flags.batch_size, shuffle=True, num_workers=4, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.flags.batch_size, num_workers=4, drop_last=True)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.flags.batch_size, num_workers=4, drop_last=True)
