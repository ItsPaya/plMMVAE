import os
import torch
import pytorch_lightning as pl
from typing import List, Optional, Sequence, Union, Any, Callable
from torch import nn, optim
from torch.nn import functional as f
from torchvision.datasets.folder import default_loader
from torchvision import transforms
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader, Dataset
import torchvision.utils as vutils
from abc import abstractmethod
from types_ import *


# model
# optimizer
# data
# training loop "magic"
# validation loop

class LightVAECelebA(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dim: List = None,
                 **kwargs) -> None:
        super(LightVAECelebA, self).__init__()
        self.save_hyperparameters()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dim is None:
            hidden_dim = [32, 64, 128, 256, 512]

        # Build encoder
        for h_dim in hidden_dim:
            modules.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels=h_dim,
                          kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU()
            )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dim[-1] * 4, latent_dim)
        self.fc_var = nn.Linear(hidden_dim[-1] * 4, latent_dim)

        # Build decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dim[-1] * 4)

        hidden_dim.reverse()

        for i in range(len(hidden_dim) - 1):
            modules.append(nn.Sequential(
                nn.ConvTranspose2d(hidden_dim[i],
                                   hidden_dim[i + 1],
                                   kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.BatchNorm2d(hidden_dim[i + 1]),
                nn.LeakyReLU()
            )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim[-1],
                               hidden_dim[-1],
                               kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dim[-1]),
            nn.LeakyReLU(),
            nn.Conv2d(hidden_dim[-1], out_channels=3,
                      kernel_size=3, padding=1),
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes input by passing through the encoder network and returns the latent codes
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """

        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # split result into mu and variance components
        # of latent gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, input: Tensor) -> Any:
        """
        Maps given latent codes
        onto image space
        :param input: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """

        result = self.decoder_input(input)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparametrize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        """
        reparameter trick to sample from N(mu, var) from N(0, 1)
        :param mu: (Tensor) Mean of latent Gaussian [B x D]
        :param log_var: (Tensor) Std of latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(.5 * log_var)
        eps = torch.randn_like(std)

        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparametrize(mu, log_var)

        return [self.decode(z), input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        """
        compute VAE loss function
        KL(N(mu, sigma), N(0, 1) = log (1/sigma) + ((sigma² + mu²)/2) - 1/2
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']  # account for the minibatch samples from the dataset
        recons_loss = f.mse_loss(recons, input)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        # was: recon + kld_w * kld_l
        # now loss == elbo
        loss = kld_weight * kld_loss - recons_loss
        # was: 'KLD': -kld_l.detach()
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    def sample(self, batch_size: int) -> Tensor:
        """
        samples from latent space and return the corresponding image space map
        :param batch_size: (Int) number of samples
        :return: (Tensor)
        """
        z = torch.randn(batch_size, self.latent_dim)

        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        results = self.forward(real_img, labels=labels)
        train_loss = self.loss_function(*results,
                                        M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                        optimizer_idx=optimizer_idx,
                                        batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, labels = batch

        results = self.forward(real_img, labels=labels)
        val_loss = self.loss_function(*results,
                                      M_N=1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                      optimizer_idx=optimizer_idx,
                                      batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

    def on_validation_end(self) -> None:
        self.sample_images()

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

        # test_input, test_label = batch
        recons = self.model.generate(test_input, labels=test_label)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels=test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):
        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    @abstractmethod
    def forward(self, *inputs: Tensor) -> Tensor:
        pass

    @abstractmethod
    def loss_function(self, *inputs: Tensor) -> Tensor:
        pass


# separate DataModule so VAE works for any Dataset
class MyCelebA(CelebA):
    """
    A workaround to address issues with pytorch's celebA dataset class
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


class VAEDataset(pl.LightningDataModule):
    """
    PyTorch Lightning data module
    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
            self,
            data_path: str,
            train_batch_size: int = 8,
            val_batch_size: int = 8,
            patch_size: Union[int, Sequence[int]] = (256, 256),
            num_workers: int = 0,
            pin_memory: bool = False,
            **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage: Optional[str] = None) -> None:
        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(), ])

        val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.CenterCrop(148),
                                             transforms.Resize(self.patch_size),
                                             transforms.ToTensor(), ])

        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=train_transforms,
            download=False,
        )

        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=val_transforms,
            download=False,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )


if __name__ == '__main()__':
    vae = LightVAECelebA()
    trainer = pl.Trainer()
    trainer.fit(vae)
