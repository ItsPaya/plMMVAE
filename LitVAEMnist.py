import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
from typing import List, Callable, Union, Any, TypeVar, Tuple
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.autograd import Variable
import pytorch_lightning as pl

Tensor = TypeVar('torch.Tensor')


class LitMNIST(pl.LightningModule):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # build encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)

        # build decoder
        modules = []
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*4)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i+1],
                                       kernel_size=3,
                                       stride=2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(nn.ConvTranspose2d(hidden_dims[-1],
                                                            hidden_dims[-1],
                                                            kernel_size=3, stride=2,
                                                            padding=1,
                                                            output_padding=1),
                                         nn.BatchNorm2d(hidden_dims[-1]),
                                         nn.LeakyReLU(),
                                         nn.Conv2d(hidden_dims[-1], out_channels=3,
                                                   kernel_size=3, padding=1),
                                         nn.Tanh()
                                         )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512, 2, 2)
        result = self.decoder(result)
        result = self.final_layer(result)

        return result

    def reparameterize(self, mu: Tensor, log_var: Tensor) -> Tensor:
        # std = torch.exp(0.5 * log_var)
        # eps = torch.randn_like(std)
        # return eps * std + mu
        std = log_var.mul(0.5).exp_()
        eps = Variable(std.data.new(std.size()).normal_())

        return eps.mul(std).add_(mu)

    def forward(self, x: Tensor, **kwargs) -> Tensor:
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)

        return [self.decode(z), x, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N']
        recons_loss = f.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_loss * kld_weight

        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'KLD': kld_loss.detach()}

    def sample(self, num_samples: int) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        samples = self.decode(z)

        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]
