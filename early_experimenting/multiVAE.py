from typing import Sequence, Union

import torch
import pytorch_lightning as pl
import torchvision.utils
from PIL.Image import Image

from torch import nn
from torch.autograd import Variable
from torch.nn import functional as f
from torch.utils.data import DataLoader
from torchvision.datasets import CelebA
from torchvision.transforms import transforms

import torch.optim as optim


# extend from BaseVAE to add to rest
# product of experts or other "magic" is missing
class MVAE(pl.LightningModule):
    def __init__(self, input_dim, latent_dim):
        super(MVAE, self).__init__()

        # encoder for modality 1
        self.encoder1 = nn.Linear(input_dim[0], latent_dim[0])
        self.mu1 = nn.Linear(latent_dim[0], latent_dim[0])
        self.logvar1 = nn.Linear(latent_dim[0], latent_dim[0])

        # encoder for modality 2
        self.encoder2 = nn.Linear(input_dim[1], latent_dim[1])
        self.mu2 = nn.Linear(latent_dim[1], latent_dim[1])
        self.logvar2 = nn.Linear(latent_dim[1], latent_dim[1])

        # decoder for modality 1
        self.decoder1 = nn.Linear(latent_dim[0], input_dim[0])

        # decoder for modality 2
        self.decoder2 = nn.Linear(latent_dim[1], input_dim[1])

        self.latents = latent_dim

    def encode(self, x1, x2):
        h1 = f.relu(self.encoder1(x1))
        mu1 = self.mu1(h1)
        logvar1 = self.logvar1(h1)

        h2 = f.relu(self.encoder2(x2))
        mu2 = self.mu2(h2)
        logvar2 = self.logvar2(h2)

        return mu1, logvar1, mu2, logvar2

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return mu + eps * std

    def decode(self, z1, z2):
        h1 = f.relu(self.decoder1(z1))
        h2 = f.relu(self.decoder2(z2))

        return h1, h2

    def forward(self, x1=None, x2=None):
        # mu1, logvar1, mu2, logvar2 = self.encode(x1, x2)
        # z1 = self.reparameterize(mu1, logvar1)
        # z2 = self.reparameterize(mu2, logvar2)
        # x1_hat, x2_hat = self.decode(z1, z2)
        # alt. only one z and two diff decoders
        mu, logvar = self.infer(x1, x2)
        z = self.reparameterize(mu, logvar)
        image_recon = self.image_decoder(z)
        attrs_recon = self.attrs_decoder(z)

        return image_recon, attrs_recon, mu, logvar

        # return x1_hat, x2_hat, mu1, logvar1, mu2, logvar2

    def productOfExperts(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps

        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var)

        return pd_mu, pd_logvar

    def prior_expert(self, size):
        mu = Variable(torch.zeros(size))
        logvar = Variable(torch.log(torch.ones(size)))

        return mu, logvar

    def swish(self, x):
        return x * f.sigmoid(x)

    def infer(self, image=None, attrs=None):
        batch_size = image.size(0) if image is None else attrs.size(0)

        mu, logvar = self.prior_expert((1, batch_size, self.latents))

        if image is not None:
            image_mu, image_logvar = self.encoder1(image)
            mu = torch.cat((mu, image_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, image_logvar.unsqueeze(0)), dim=0)

        if attrs is not None:
            attrs_mu, attrs_logvar = self.encoder2(attrs)
            mu = torch.cat((mu, attrs_mu.unsqueeze(0)), dim=0)
            logvar = torch.cat((logvar, attrs_logvar.unsqueeze(0)), dim=0)

        mu, logvar = self.productOfExperts(mu, logvar)

        return mu, logvar

    def training_step(self, batch, batch_idx):
        img, attrs = batch
        recon_img_1, recon_attrs_1, mu_1, logvar_1 = MVAE(img, attrs)
        recon_img_2, recon_attrs_2, mu_2, logvar_2 = MVAE(img)
        recon_img_3, recon_attrs_3, mu_3, logvar_3 = MVAE(x2=attrs)

        joint_loss = self.loss_function(recon_img_1, img, recon_attrs_1, attrs, mu_1, logvar_1)
        img_loss = self.loss_function(recon_img_2, img, None, None, mu_2, logvar_2)
        attrs_loss = self.loss_function(None, None, recon_attrs_3, attrs, mu_3, logvar_3)
        train_loss = joint_loss + img_loss + attrs_loss

        log = {'train_loss': train_loss}

        return {'loss': train_loss, 'log': log}

    def validation_step(self, batch, batch_idx):
        img, attrs = batch
        recon_img_1, recon_attrs_1, mu_1, logvar_1 = MVAE(img, attrs)
        recon_img_2, recon_attrs_2, mu_2, logvar_2 = MVAE(img)
        recon_img_3, recon_attrs_3, mu_3, logvar_3 = MVAE(x2=attrs)

        joint_loss = self.loss_function(recon_img_1, img, recon_attrs_1, attrs, mu_1, logvar_1)
        img_loss = self.loss_function(recon_img_2, img, None, None, mu_2, logvar_2)
        attrs_loss = self.loss_function(None, None, recon_attrs_3, attrs, mu_3, logvar_3)
        val_loss = joint_loss + img_loss + attrs_loss

        return {'val_loss': val_loss}

    def validation_epoch_end(self, outputs):
        val_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        log = {'avg_val_loss': val_loss}

        return {'log': log}

    def loss_function(self, recon_img, image, recon_attrs, attrs, mu, logvar, lambda_img=1.,
                      lambda_attrs=1., annealing_factor=1):
        # recons = args[0], input = args[1], mu = args[2], logvar = args[3]
        img_bce, attrs_bce = 0, 0  # default

        if recon_img is not None and image is not None:
            img_bce = torch.sum(self.binary_cross_entropy_with_logits(
                recon_img.view(-1, 3 * 64 * 64),
                image.view(-1, 3 * 64 * 64)
            ), dim=1)

        if recon_attrs is not None and attrs is not None:
            for i in range(self.latents):
                attr_bce = self.binary_cross_entropy_with_logits(
                    recon_attrs[:, i], attrs[:, i]
                )
                attrs_bce += attr_bce
        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        elbo = torch.mean(lambda_img * img_bce + lambda_attrs * attr_bce + annealing_factor * kld)

        return elbo

    def binary_cross_entropy_with_logits(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})").__format__(
                target.size(), input.size()
            )

        return (torch.clamp(input, 0) - input * target
                + torch.log(1 + torch.exp(-torch.abs(input))))

    def configure_optimizers(self):
        optimizer = optim.Adam(MVAE.parameters(), lr=1e-2)

        return optimizer


class CelebADataset(pl.LightningDataModule):
    def __init__(self,
                 data_path: str,
                 train_batch_size: int = 8,
                 val_batch_size: int = 8,
                 patch_size: Union[int, Sequence[int]] = (256, 256),
                 num_workers: int = 0,
                 pin_memory: bool = False,
                 **kwargs
                 ):
        super(CelebADataset, self).__init__()
        self.data_dir = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                    transforms.CenterCrop(148),
                                                    transforms.Resize(self.patch_size),
                                                    transforms.ToTensor(), ])

        self.val_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                                  transforms.CenterCrop(148),
                                                  transforms.Resize(self.patch_size),
                                                  transforms.ToTensor(), ])

    def prepare_data(self):
        self.train_dataset = MyCelebA(
            self.data_dir,
            split='train',
            transform=self.train_transforms,
            download=False,
        )

        # Replace CelebA with your dataset
        self.val_dataset = MyCelebA(
            self.data_dir,
            split='test',
            transform=self.val_transforms,
            download=False,
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers,
                          shuffle=True, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=144, num_workers=self.num_workers,
                          shuffle=True, pin_memory=self.pin_memory)


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.

    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """

    def _check_integrity(self) -> bool:
        return True


if __name__ == '__main__':
    vae = MVAE([3, 3], [128, 128])
    trainer = pl.Trainer(fast_dev_run=True)
    trainer.fit(vae)
