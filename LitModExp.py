import os
from itertools import chain, combinations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim

from MNISTSVHNTEXT.ConvNetImgMNIST import EncoderImg, DecoderImg
from MNISTSVHNTEXT.ConvNetImgSVHN import EncoderSVHN, DecoderSVHN
from MNISTSVHNTEXT.ConvNetTextMNIST import EncoderText, DecoderText
from MNISTSVHNTEXT.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from MNISTSVHNTEXT.ConvNetImgClfMNIST import ClfImg as ClfImgMNIST
from MNISTSVHNTEXT.ConvNetImgClfSVHN import ClfImgSVHN
from MNISTSVHNTEXT.ConvNetTextClf import ClfText as ClfText
from modalities.MNIST import MNIST
from modalities.SVHN import SVHN
from modalities.Text import Text


class MultiModVAE(pl.LightningModule):
    def __init__(self, config, flags):
        super().__init__()
        self.config = config
        self.flags = flags
        self.optimizer = None
        self.objective = None
        self.subsets = self.set_subsets()
        self.mod_names = self.get_mod_names()
        self.num_mods = len(list(self.mod_names.keys()))
        self.model = self.get_model()

    def get_mod_names(self):
        mod_names = {}
        for i, m in enumerate(self.config.mods):
            mod_names['mod_{}'.format(i+1)] = m['mod_type']

        return mod_names

    def set_clfs(self):
        model_clf_m1 = None
        model_clf_m2 = None
        model_clf_m3 = None
        if self.flags.use_clf:
            model_clf_m1 = ClfImgMNIST()
            model_clf_m1.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m1)))

            model_clf_m2 = ClfImgSVHN()
            model_clf_m2.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m2)))

            model_clf_m3 = ClfText(self.flags)
            model_clf_m3.load_state_dict(torch.load(os.path.join(self.flags.dir_clf,
                                                                 self.flags.clf_save_m3)))

        clfs = {'mnist': model_clf_m1,
                'svhn': model_clf_m2,
                'text': model_clf_m3}
        return clfs

    def set_modalities(self):
        mod1 = MNIST('mnist', EncoderImg(self.flags), DecoderImg(self.flags),
                    self.flags.class_dim, self.flags.style_m1_dim, 'laplace')
        mod2 = SVHN('svhn', EncoderSVHN(self.flags), DecoderSVHN(self.flags),
                    self.flags.class_dim, self.flags.style_m2_dim, 'laplace',
                    self.plot_img_size)
        mod3 = Text('text', EncoderText(self.flags), DecoderText(self.flags),
                    self.flags.class_dim, self.flags.style_m3_dim, 'categorical',
                    self.flags.len_sequence,
                    self.alphabet,
                    self.plot_img_size,
                    self.font)
        mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}
        return mods

    def set_subsets(self):
        """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
        xs = list(self.modalities)
        # note we return an iterator rather than a list
        subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                           range(len(xs) + 1))
        subsets = dict()
        for k, mod_names in enumerate(subsets_list):
            mods = []
            for l, mod_name in enumerate(sorted(mod_names)):
                mods.append(self.modalities[mod_name])
            key = '_'.join(sorted(mod_names))
            subsets[key] = mods
            subsets.pop('', None)
        return subsets

    def get_model(self):
        model = VAEtrimodalSVHNMNIST(self.flags, self.config.mods, self.subsets)
        return model

    def training_step(self, batch, batch_idx):
        loss = self.model.objective(batch)
        for key in loss.keys():
            if key != 'recon_loss':
                self.log('val_{}'.format(key), loss[key].sum(), batch_size=self.config.batch_size)
            else:
                for i, p_l in enumerate(loss[key]):
                    self.log('Mod_{}_TrainLoss'.format(i), p_l.sum(), batch_size=self.config.batch_size)
        return loss['loss']

    def validation_step(self, batch, batch_idx):
        loss = self.model.objective(batch)
        for key in loss.keys():
            if key != 'recon_loss':
                self.log('val_{}'.format(key), loss[key].sum(), batch_size=self.config.batch_size)
            else:
                for i, p_l in enumerate(loss[key]):
                    self.log('Mod_{}_ValLoss'.format(i), p_l.sum(), batch_size=self.config.batch_size)
        return loss['val_loss']

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=self.config.lr,
            betas=(self.config.beta_1, self.config.beta_2))
        return optimizer
