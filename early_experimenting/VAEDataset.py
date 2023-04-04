import os
import torch
import pytorch_lightning as pl
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import zipfile


class MNISTDataModule(pl.LightningDataModule):
    """
    Args:
        data_dir: root directory of your dataset
        train_batch_size: batch_size during training
        val_batch_size: batch_size during validation
        patch_size: size of the crop to take from the original img
        num_workers: number of parallel workers to create to load data items
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. can improve performance on GPUs
    """

    def __init__(self,
                 data_path: str,
                 train_batch_size: int = 8,
                 val_batch_size: int = 8,
                 patch_size: Union[int, Sequence[int]] = (256, 256),
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 **kwargs):
        super().__init__()

        self.val_dataset = None
        self.train_dataset = None
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    @property
    def num_classes(self) -> int:
        return 10

    def prepare_data(self) -> None:
        self.train_dataset = MNIST('MNIST', train=True, transform=transforms.ToTensor(), download=False)
        self.val_dataset = MNIST('MNIST', train=False, transform=transforms.ToTensor(), download=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=self.pin_memory
                          )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset,
                          batch_size=self.val_batch_size,
                          num_workers=self.num_workers,
                          shuffle=False,
                          pin_memory=self.pin_memory
                          )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(self.val_dataset,
                          batch_size=144,
                          num_workers=self.num_workers,
                          shuffle=True,
                          pin_memory=self.pin_memory
                          )
