import torch
import torch.nn as nn
import pytorch_lightning as pl

from utils.utils import Flatten, Unflatten


class EncoderImg(pl.LightningModule):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, config):
        super(EncoderImg, self).__init__()

        self.config = config
        self.shared_encoder = nn.Sequential(                          # input shape (3, 28, 28)
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),     # -> (32, 14, 14)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),    # -> (64, 7, 7)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),   # -> (128, 4, 4)
            nn.ReLU(),
            Flatten(),                                                # -> (2048)
            nn.Linear(2048, config.style_dim + config.class_dim),       # -> (ndim_private + ndim_shared)
            nn.ReLU(),
        )

        # content branch
        self.class_mu = nn.Linear(config.style_dim + config.class_dim, config.class_dim)
        self.class_logvar = nn.Linear(config.style_dim + config.class_dim, config.class_dim)
        # optional style branch
        if self.config.method_mods['factorized_representation']:
            self.style_mu = nn.Linear(config.style_dim + config.class_dim, config.style_dim)
            self.style_logvar = nn.Linear(config.style_dim + config.class_dim, config.style_dim)

    def forward(self, x):
        h = self.shared_encoder(x)
        if self.config.method_mods['factorized_representation']:
            return self.style_mu(h), self.style_logvar(h), self.class_mu(h), \
                   self.class_logvar(h)
        else:
            return None, None, self.class_mu(h), self.class_logvar(h)


class DecoderImg(pl.LightningModule):
    """
    Adopted from:
    https://www.cs.toronto.edu/~lczhang/360/lec/w05/autoencoder.html
    """
    def __init__(self, config):
        super(DecoderImg, self).__init__()
        self.config = config
        self.decoder = nn.Sequential(
            nn.Linear(config.style_dim + config.class_dim, 2048),                                # -> (2048)
            nn.ReLU(),
            Unflatten((128, 4, 4)),                                                            # -> (128, 4, 4)
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1),                   # -> (64, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),  # -> (32, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=2, padding=1, output_padding=1),   # -> (3, 28, 28)
        )

    def forward(self, style_latent_space, class_latent_space):
        if self.config.method_mods['factorized_representation']:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
        else:
            z = class_latent_space
        x_hat = self.decoder(z)
        # x_hat = torch.sigmoid(x_hat)
        return x_hat, torch.tensor(0.75).to(z.device)  # NOTE: consider learning scale param, too
