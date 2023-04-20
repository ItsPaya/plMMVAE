import pytorch_lightning as pl

from torch.utils.data import DataLoader, random_split
from torchvision import transforms as transforms

from MMNIST.MMNISTDataset import MMNISTDataset


class MMNISTDataModule(pl.LightningDataModule):
    def __init__(self, config, alphabet):
        super(MMNISTDataModule, self).__init__()
        self.config = config
        self.alphabet = alphabet
        self.dataset_train = None
        self.dataset_test = None
        self.dataset_val = None
        self.batch_size = config.batch_size
        self.num_workers = config.num_workers
        self.drop_last = config.drop_last

    def setup(self, stage=None):
        print(len(self.config.unimodal_datapaths['train']))
        transform = transforms.Compose([transforms.ToTensor()])
        mmnist = MMNISTDataset(self.config.unimodal_datapaths['train'], transform=transform)
        train_split = int(.8 * len(mmnist))
        val_split = len(mmnist) - train_split
        self.dataset_train, self.dataset_val = random_split(mmnist, [train_split, val_split])
        self.dataset_test = MMNISTDataset(self.config.unimodal_datapaths['test'], transform=transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size,
                          num_workers=self.num_workers, drop_last=self.drop_last)
