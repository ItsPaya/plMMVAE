import torch
import torch.nn as nn
import pytorch_lightning as pl


class FeatureEncText(pl.LightningModule):
    def __init__(self, dim, num_features):
        super(FeatureEncText, self).__init__()
        self.dim = dim
        self.conv1 = nn.Conv1d(num_features, 2*self.dim, kernel_size=1)
        self.conv2 = nn.Conv1d(2*self.dim, 2*self.dim, kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv5 = nn.Conv1d(2*self.dim, 2*self.dim, kernel_size=4, stride=2, padding=0, dilation=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.transpose(-2, -1)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv5(out)
        out = self.relu(out)
        h = out.view(-1, 2*self.dim)
        return h


class EncoderText(pl.LightningModule):
    def __init__(self, config):
        super(EncoderText, self).__init__()
        self.config = config
        self.text_feature_enc = FeatureEncText(config.dim, config.num_features)
        if config.method_mods['factorized_representation']:
            # style
            self.style_mu = nn.Linear(in_features=2*config.dim, out_features=config.mods[3]['style_dim'], bias=True)
            self.style_logvar = nn.Linear(in_features=2*config.dim, out_features=config.mods[3]['style_dim'], bias=True)
            # class
            self.class_mu = nn.Linear(in_features=2*config.dim, out_features=config.class_dim, bias=True)
            self.class_logvar = nn.Linear(in_features=2*config.dim, out_features=config.class_dim, bias=True)
        else:
            #non-factorized
            self.latent_mu = nn.Linear(in_features=2*config.dim, out_features=config.class_dim, bias=True)
            self.latent_logvar = nn.Linear(in_features=2*config.dim, out_features=config.class_dim, bias=True)

    def forward(self, x):
        h = self.text_feature_enc(x)
        if self.config.method_mods['factorized_representation']:
            style_latent_space_mu = self.style_mu(h)
            style_latent_space_logvar = self.style_logvar(h)
            class_latent_space_mu = self.class_mu(h)
            class_latent_space_logvar = self.class_logvar(h)
            return style_latent_space_mu, style_latent_space_logvar, class_latent_space_mu, class_latent_space_logvar
        else:
            latent_space_mu = self.latent_mu(h)
            latent_space_logvar = self.latent_logvar(h)
            return None, None, latent_space_mu, latent_space_logvar


class DecoderText(pl.LightningModule):
    def __init__(self, config):
        super(DecoderText, self).__init__()
        self.config = config
        if config.method_mods['factorized_representation']:
            self.linear_factorized = nn.Linear(config.mods[3]['style_dim']+config.class_dim,
                                               2*config.dim)
        else:
            self.linear = nn.Linear(config.class_dim, 2*config.dim)
        self.conv1 = nn.ConvTranspose1d(2*config.dim, 2*config.dim,
                                        kernel_size=4, stride=1, padding=0, dilation=1)
        self.conv2 = nn.ConvTranspose1d(2*config.dim, 2*config.dim,
                                        kernel_size=4, stride=2, padding=1, dilation=1)
        self.conv_last = nn.Conv1d(2*config.dim, config.num_features, kernel_size=1)
        self.relu = nn.ReLU()
        self.out_act = nn.LogSoftmax(dim=-2)

    def forward(self, style_latent_space, class_latent_space):
        if self.config.method_mods['factorized_representation']:
            z = torch.cat((style_latent_space, class_latent_space), dim=1)
            z = self.linear_factorized(z)
        else:
            z = self.linear(class_latent_space)
        x_hat = z.view(z.size(0), z.size(1), 1)
        x_hat = self.conv1(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv2(x_hat)
        x_hat = self.relu(x_hat)
        x_hat = self.conv_last(x_hat)
        log_prob = self.out_act(x_hat)
        log_prob = log_prob.transpose(-2, -1)
        return [log_prob]
