import os

import random
from abc import ABC

import numpy as np

import PIL.Image as Image
from PIL import ImageFont

import torch
from torchvision import transforms
import torch.optim as optim
from sklearn.metrics import accuracy_score

import utils
from modalities.MNIST import MNIST
from modalities.SVHN import SVHN
from modalities.Text import Text

from SVHNMNISTDataset import SVHNMNIST
from VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from ConvNetImgClfMNIST import ClfImg as ClfImgMNIST
from ConvNetImgClfSVHN import ClfImgSVHN
from ConvNetTextClf import ClfText as ClfText

from ConvNetImgMNIST import EncoderImg, DecoderImg
from ConvNetImgSVHN import EncoderSVHN, DecoderSVHN
from ConvNetTextMNIST import EncoderText, DecoderText

from utils.BaseExperiment import BaseExperiment


class MNISTSVHNText(BaseExperiment, ABC):
    def __init__(self, flags, alphabet):
        super().__init__(flags)
        self.plot_img_size = torch.Size((3, 28, 28))
        self.font = ImageFont.truetype('FreeSerif.ttf', 38)
        self.alphabet = alphabet
        self.flags.num_features = len(alphabet)

        self.modalities = self.set_modalities()
        self.num_modalities = len(self.modalities.keys())
        self.subsets = self.set_subsets()
        self.dataset_train = None
        self.dataset_val = None
        self.dataset_test = None
        self.set_dataset()

        self.mm_vae = self.set_model()
        self.clfs = self.set_clfs()
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = self.get_test_samples()
        self.eval_metric = accuracy_score
        self.paths_fid = self.set_paths_fid()

        self.labels = ['digit']

    def set_model(self):
        model = VAEtrimodalSVHNMNIST(self.flags, self.modalities, self.subsets)
        return model

    def training_step(self, batch, batch_idx):
        beta_style = self.flags.beta_style
        beta_content = self.flags.beta_content
        beta = self.flags.beta
        rec_weight = 1.0

        mm_vae = self.mm_vae
        batch_d = batch[0]
        batch_l = batch[1]
        mods = self.modalities
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key]
        results = mm_vae(batch_d)

        log_probs, weighted_log_prob = self.calc_log_probs(self, results, batch)
        group_divergence = results['joint_divergence']

        klds = self.calc_klds(self, results)
        if self.flags.factorized_representation:
            klds_style = self.calc_klds_style(self, results)

        if (self.flags.modality_jsd or self.flags.modality_moe
                or self.flags.joint_elbo):
            if self.flags.factorized_representation:
                kld_style = self.calc_style_kld(self, klds_style)
            else:
                kld_style = 0.0
            kld_content = group_divergence
            kld_weighted = beta_style * kld_style + beta_content * kld_content
            total_loss = rec_weight * weighted_log_prob + beta * kld_weighted
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
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      self.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(self, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
            elbo_joint = utils.calc_elbo(self, 'joint', log_probs, klds_joint)
            elbos['joint'] = elbo_joint
            total_loss = sum(elbos.values())

        out_basic_routine = dict()
        out_basic_routine['results'] = results
        out_basic_routine['log_probs'] = log_probs
        out_basic_routine['total_loss'] = total_loss
        out_basic_routine['klds'] = klds
        return out_basic_routine

    def validation_step(self, batch, batch_idx):
        beta_style = self.flags.beta_style
        beta_content = self.flags.beta_content
        beta = self.flags.beta
        rec_weight = 1.0

        mm_vae = self.mm_vae
        batch_d = batch[0]
        batch_l = batch[1]
        mods = self.modalities
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = batch_d[m_key]
        results = mm_vae(batch_d)

        log_probs, weighted_log_prob = self.calc_log_probs(self, results, batch)
        group_divergence = results['joint_divergence']

        klds = self.calc_klds(self, results)
        if self.flags.factorized_representation:
            klds_style = self.calc_klds_style(self, results)

        if (self.flags.modality_jsd or self.flags.modality_moe
                or self.flags.joint_elbo):
            if self.flags.factorized_representation:
                kld_style = self.calc_style_kld(self, klds_style)
            else:
                kld_style = 0.0
            kld_content = group_divergence
            kld_weighted = beta_style * kld_style + beta_content * kld_content
            total_loss = rec_weight * weighted_log_prob + beta * kld_weighted
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
                    r_mod = mm_vae(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      self.flags.batch_size)
                    log_prob = {m_key: log_prob_mod}
                    klds_mod = {'content': klds[m_key],
                                'style': {m_key: kld_style_m}}
                    elbo_mod = utils.calc_elbo(self, m_key, log_prob, klds_mod)
                    elbos[m_key] = elbo_mod
            elbo_joint = utils.calc_elbo(self, 'joint', log_probs, klds_joint)
            elbos['joint'] = elbo_joint
            total_loss = sum(elbos.values())

        out_basic_routine = dict()
        out_basic_routine['results'] = results
        out_basic_routine['log_probs'] = log_probs
        out_basic_routine['total_loss'] = total_loss
        out_basic_routine['klds'] = klds
        return out_basic_routine

    def calc_log_probs(self, result, batch):
        mods = self.modalities
        log_probs = dict()
        weighted_log_prob = 0.0
        for m, m_key in enumerate(mods.keys()):
            mod = mods[m_key]
            log_probs[mod.name] = -mod.calc_log_prob(result['rec'][mod.name],
                                                     batch[0][mod.name],
                                                     self.flags.batch_size)
            weighted_log_prob += self.rec_weights[mod.name] * log_probs[mod.name]
        return log_probs, weighted_log_prob

    def calc_klds(self, result):
        latents = result['latents']['subsets']
        klds = dict()
        for m, key in enumerate(latents.keys()):
            mu, logvar = latents[key]
            klds[key] = self.calc_kl_divergence(mu, logvar,
                                           norm_value=self.flags.batch_size)
        return klds

    def calc_klds_style(self, result):
        latents = result['latents']['modalities']
        klds = dict()
        for m, key in enumerate(latents.keys()):
            if key.endswith('style'):
                mu, logvar = latents[key]
                klds[key] = self.calc_kl_divergence(mu, logvar,
                                               norm_value=self.flags.batch_size)
        return klds

    def calc_style_kld(self, klds):
        mods = self.modalities
        style_weights = self.style_weights
        weighted_klds = 0.0
        for m, m_key in enumerate(mods.keys()):
            weighted_klds += style_weights[m_key] * klds[m_key + '_style']
        return weighted_klds

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

    def configure_optimizers(self):
        # optimizer definition
        optimizer = optim.Adam(
            list(self.mm_vae.parameters()),
            lr=self.flags.initial_learning_rate,
            betas=(self.flags.beta_1, self.flags.beta_2))
        return optimizer

    def set_rec_weights(self):
        rec_weights = dict()
        ref_mod_d_size = self.modalities['svhn'].data_size.numel()
        for k, m_key in enumerate(self.modalities.keys()):
            mod = self.modalities[m_key]
            numel_mod = mod.data_size.numel()
            rec_weights[mod.name] = float(ref_mod_d_size/numel_mod)
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
