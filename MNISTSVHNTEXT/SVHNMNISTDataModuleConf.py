import PIL.Image
import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms

from MNISTSVHNTEXT.SVHNMNISTDataset import SVHNMNIST


class SVHNMNISTDataModuleC(pl.LightningDataModule):
    def __init__(self, config, alphabet):
        super(SVHNMNISTDataModuleC, self).__init__()
        self.config = config
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
        # print(len(svhnmnist))
        train_split = int(.8 * len(svhnmnist))
        val_split = len(svhnmnist) - train_split
        train, val = random_split(svhnmnist, [train_split, val_split])
        test = SVHNMNIST(self.config,
                         self.alphabet,
                         train=False,
                         transform=transforms)
        self.dataset_train = train
        self.dataset_val = val
        self.dataset_test = test

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)
