import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim

from MNISTSVHNTEXT import VAEtrimodalSVHNMNIST
from utils.BaseMMVae import BaseMMVae


class LitModule(BaseMMVae):
    def __init__(self, flags, modalities, subsets):
        super().__init__(flags, modalities, subsets)

    # def forward(self, x):
        # return

    def training_step(self, batch, batch_idx):
        latents = self.inference(batch)
        results = dict()
        results['latents'] = latents
        results['group_distr'] = latents['joint']
        class_embeddings = self.reparameterize(latents['joint'][0],
                                               latents['joint'][1])
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights'])
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        results_rec = dict()
        enc_mods = latents['modalities']
        for m, m_key in enumerate(self.modalities.keys()):
            if m < len(batch):
                m_s_mu, m_s_logvar = enc_mods[m_key + '_style']
                if self.flags.factorized_representation:
                    m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar)
                else:
                    m_s_embeddings = None
                m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings))
                results_rec[m_key] = m_rec
        results['rec'] = results_rec

        return results

    def validation_step(self, batch, batch_idx):
        latents = self.inference(batch)
        results = dict()
        results['latents'] = latents
        results['group_distr'] = latents['joint']
        class_embeddings = self.reparameterize(latents['joint'][0],
                                               latents['joint'][1])
        div = self.calc_joint_divergence(latents['mus'],
                                         latents['logvars'],
                                         latents['weights'])
        for k, key in enumerate(div.keys()):
            results[key] = div[key]

        results_rec = dict()
        enc_mods = latents['modalities']
        for m, m_key in enumerate(self.modalities.keys()):
            if m < len(batch):
                m_s_mu, m_s_logvar = enc_mods[m_key + '_style']
                if self.flags.factorized_representation:
                    m_s_embeddings = self.reparameterize(mu=m_s_mu, logvar=m_s_logvar)
                else:
                    m_s_embeddings = None
                m_rec = self.lhoods[m_key](*self.decoders[m_key](m_s_embeddings, class_embeddings))
                results_rec[m_key] = m_rec
        results['rec'] = results_rec

        return results

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        return optimizer
