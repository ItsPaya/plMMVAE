import os
import random
from itertools import chain, combinations

import numpy as np
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
from utils import utils


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

    def set_rec_weights(self):
        rec_weights = dict()
        ref_mod_d_size = self.modalities['svhn'].data_size.numel()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
        return rec_weights

    def set_style_weights(self):
        weights = dict()
        weights['mnist'] = self.flags.beta_m1_style
        weights['svhn'] = self.flags.beta_m2_style
        weights['text'] = self.flags.beta_m3_style
        return weights

    def get_test_samples(self, num_images=10):
        n_test = self.dataset_test.__len__()
        samples = []
        for i in range(num_images):
            while True:
                sample, target = self.dataset_test.__getitem__(random.randint(0, n_test))
                if target == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key]
                    samples.append(sample)
                    break
        return samples

    def mean_eval_metric(self, values):
        return np.mean(np.array(values))

    def get_prediction_from_attr(self, attr, index=None):
        pred = np.argmax(attr, axis=1).astype(int)
        return pred

    def eval_label(self, values, labels, index):
        pred = self.get_prediction_from_attr(values)
        return self.eval_metric(labels, pred)

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

    def basic_routine(self, batch):
        beta_style = self.flags.beta_style
        beta_content = self.flags.beta_content
        beta = self.flags.beta
        rec_weight = 1.0

        batch_d = batch[0]
        batch_l = batch[1]
        mods = self.modalities

        results = self.forward(batch_d)

        log_probs, weighted_log_prob = self.calc_log_probs(results, batch)
        group_divergence = results['joint_divergence']

        klds = self.calc_klds(results)
        if self.flags.factorized_representation:
            klds_style = self.calc_klds_style(results)

        if (self.flags.modality_jsd or self.flags.modality_moe or
                self.flags.joint_elbo):
            if self.flags.factorized_representation:
                kld_style = self.calc_style_kld(klds_style)
            else:
                kld_style = 0.0
            kld_content = group_divergence
            kld_weighted = beta_style * kld_style + beta_content * kld_content
            total_loss = rec_weight * weighted_log_prob + beta - kld_weighted
        elif self.flags.modality_poe:
            klds_joint = {'content': group_divergence,
                          'style': dict()}
            elbos = dict()
            for m, m_key in enumerate(mods.keys()):
                mod = mods[m_key]
                if self.flags.factorized_representation:
                    kld_style_m = klds_style[m_key + '_style']
                else:
                    kld_style_m = 0.0
                klds_joint['style'][m_key] = kld_style_m
                if self.flags.poe_unimodal_elbos:
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = self.forward(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      self.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.utils.calc_elbo(self, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
            elbo_joint = utils.utils.calc_elbo(self, 'joint', log_probs, klds_joint)
            elbos['joint'] = elbo_joint
            total_loss = sum(elbos.values())

        out_basic_routine = dict()
        out_basic_routine['results'] = results
        out_basic_routine['log_probs'] = log_probs
        out_basic_routine['loss'] = total_loss
        out_basic_routine['klds'] = klds

        return out_basic_routine

    def training_step(self, batch, batch_idx):
        basic_routine = self.basic_routine(batch)
        results = basic_routine['results']
        total_loss = basic_routine['loss']
        klds = basic_routine['klds']
        log_probs = basic_routine['log_probs']

        latents = results['latents']
        l_mods = latents['modalities']

        self.log_dict({'train_loss': total_loss.data.item(), 'train_log_probs': log_probs,
                       'train_klds': klds}, on_epoch=True)
        self.log('train_group_divergence', {'group_div': results['joint_divergence'].item()}, on_epoch=True)
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.log_dict({'train_mu': l_mods[key][0].mean().item()}, on_epoch=True)
            if not l_mods[key][1] is None:
                self.log('train_logvar', l_mods[key][1].mean().item(), on_epoch=True)
        # self.logger.write_training_logs(results, total_loss, log_probs, klds)

        return basic_routine

    def validation_step(self, batch, batch_idx):
        basic_routine = self.basic_routine(batch)
        results = basic_routine['results']
        total_loss = basic_routine['loss']
        klds = basic_routine['klds']
        log_probs = basic_routine['log_probs']

        # self.log_dict({'val_results': results, 'bal_tot_loss': total_loss, 'val_klds': klds,
        # 'val_log_probs': log_probs}, on_epoch=True)

        latents = results['latents']
        l_mods = latents['modalities']

        self.log_dict({'val_loss': total_loss.data.item(), 'val_log_probs': log_probs,
                       'val_klds': klds}, on_epoch=True)
        self.log('val_group_divergence', {'group_div': results['joint_divergence'].item()}, on_epoch=True)
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.log_dict({'val_mu': l_mods[key][0].mean().item()}, on_epoch=True)
            if not l_mods[key][1] is None:
                self.log('val_logvar', l_mods[key][1].mean().item(), on_epoch=True)

        return basic_routine

    def test_step(self, batch, batch_idx):
        basic_routine = self.basic_routine(batch)
        results = basic_routine['results']
        total_loss = basic_routine['loss']
        klds = basic_routine['klds']
        log_probs = basic_routine['log_probs']

        # self.log_dict({'val_results': results, 'bal_tot_loss': total_loss, 'val_klds': klds,
        # 'val_log_probs': log_probs}, on_epoch=True)
        latents = results['latents']
        l_mods = latents['modalities']

        self.log_dict({'test_loss': total_loss}, on_epoch=True)
        self.log_dict({'ts_log_probs': log_probs, 'ts_klds': klds}, on_epoch=True)
        self.log_dict({'ts_group_div': results['joint_divergence'].item()}, on_epoch=True)
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.log_dict({'ts_mu': l_mods[key][0].mean().item()}, on_epoch=True)
            if not l_mods[key][1] is None:
                self.log('ts_logvar', l_mods[key][1].mean().item(), on_epoch=True)
        #  self.logger.write_testing_logs(results, total_loss, log_probs, klds)

        return basic_routine

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=self.config.lr,
            betas=(self.config.beta_1, self.config.beta_2))
        return optimizer
