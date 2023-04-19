import os
import random
from abc import ABC
from itertools import chain, combinations

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import accuracy_score

from divergence_measures.kl_div import calc_kl_divergence
from eval_metrics.coherence import test_generation
from eval_metrics.likelihood import estimate_likelihoods
from eval_metrics.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from eval_metrics.sample_quality import calc_prd_score
from plotting import generate_plots

from MNISTSVHNTEXT.ConvNetImgMNIST import EncoderImg, DecoderImg
from MNISTSVHNTEXT.ConvNetImgSVHN import EncoderSVHN, DecoderSVHN
from MNISTSVHNTEXT.ConvNetTextMNIST import EncoderText, DecoderText
from MMNIST.ConvNetworksImgCMNIST import EncoderImg as EncImgCMNIST
from MMNIST.ConvNetworksImgCMNIST import DecoderImg as DecImgCMNIST

from MNISTSVHNTEXT.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from MMNIST.VAEMMNIST import VAEMMNIST

from MNISTSVHNTEXT.ConvNetImgClfMNIST import ClfImg as ClfImgMNIST
from MNISTSVHNTEXT.ConvNetImgClfSVHN import ClfImgSVHN
from MNISTSVHNTEXT.ConvNetTextClf import ClfText as ClfText
from MMNIST.ConvNetworkImgClfCMNIST import ClfImg as ClfImgCMNIST

from modalities.MNIST import MNIST
from modalities.SVHN import SVHN
from modalities.Text import Text
from modalities.CMNIST import CMNIST
from utils import utils


class MultiModVAE(pl.LightningModule, ABC):
    def __init__(self, config, font, plot_img_size, alphabet):
        super().__init__()
        self.config = config
        self.plot_img_size = plot_img_size
        self.font = font
        self.alphabet = alphabet
        self.config.num_features = len(alphabet)

        self.modalities = self.set_modalities()  # get from config
        self.subsets = self.set_subsets()
        self.num_modalities = len(self.modalities.keys())

        self.mm_vae = self.get_model()  # get from config
        self.clfs = self.set_clfs()  # get from config
        self.optimizer = None
        self.rec_weights = self.set_rec_weights()
        self.style_weights = self.set_style_weights()

        self.test_samples = None
        self.eval_metric = accuracy_score
        self.paths_fid = self.set_paths_fid()

        self.labels = ['digit']

    def get_model(self):
        # make it to choose model via config? with provided params
        if self.config.dataset == 'MMNIST':
            model = VAEMMNIST(self.config, self.modalities, self.subsets)
        else:
            model = VAEtrimodalSVHNMNIST(self.config, self.modalities, self.subsets)
        model = model.to(self)
        return model

    def training_step(self, batch, batch_idx):
        basic_routine = self.basic_routine(batch)
        results = basic_routine['results']
        total_loss = basic_routine['loss']
        klds = basic_routine['klds']
        log_probs = basic_routine['log_probs']

        latents = results['latents']
        l_mods = latents['modalities']

        self.log('train_loss', total_loss.data.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict({'train_log_probs': log_probs,
                       'train_klds': klds}, on_epoch=True, on_step=True)
        self.log('train_group_divergence', {'group_div': results['joint_divergence'].item()},
                 on_epoch=True, on_step=True)
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.log_dict({'train_mu': l_mods[key][0].mean().item()}, on_epoch=True, on_step=True)
            if not l_mods[key][1] is None:
                self.log('train_logvar', l_mods[key][1].mean().item(), on_epoch=True, on_step=True)
        # self.logger.write_training_logs(results, total_loss, log_probs, klds)

        return basic_routine['loss']

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

        self.log('val_loss', total_loss.data.item(), prog_bar=True, on_step=True, on_epoch=True)
        self.log_dict({'val_log_probs': log_probs,
                       'val_klds': klds}, on_epoch=True, on_step=True)
        self.log('val_group_divergence', results['joint_divergence'].item(),
                 on_epoch=True, on_step=True, prog_bar=True)
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.log_dict({'val_mu': l_mods[key][0].mean().item()}, on_epoch=True, on_step=True)
            if not l_mods[key][1] is None:
                self.log('val_logvar', l_mods[key][1].mean().item(), on_epoch=True, on_step=True)

        return basic_routine['loss']

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

        self.log_dict({'test_loss': total_loss}, on_epoch=True, on_step=True, prog_bar=True)
        self.log_dict({'ts_log_probs': log_probs, 'ts_klds': klds}, on_epoch=True, on_step=True)
        self.log_dict({'ts_group_div': results['joint_divergence'].item()},
                      on_epoch=True, on_step=True, prog_bar=True)
        for k, key in enumerate(l_mods.keys()):
            if not l_mods[key][0] is None:
                self.log_dict({'ts_mu': l_mods[key][0].mean().item()}, on_epoch=True, on_step=True)
            if not l_mods[key][1] is None:
                self.log('ts_logvar', l_mods[key][1].mean().item(), on_epoch=True, on_step=True)

        return basic_routine['loss']

    def on_test_epoch_end(self):
        self.test_samples = self.get_test_samples()
        epoch = self.current_epoch
        plots = generate_plots(self, epoch)
        for k, p_key in enumerate(plots.keys()):
            ps = plots[p_key]
            for l, name in enumerate(ps.keys()):
                fig = ps[name]
                self.logger.experiment.add_image(p_key + '_' + name,
                                                 fig, epoch, dataformats='HWC')

        if (epoch + 1) % self.config.evaluation['eval_freq'] == 0 or\
                (epoch + 1) == self.config.trainer_params['max_epoch']:
            if self.config.evaluation['eval_lr']:
                clf_lr = train_clf_lr_all_subsets(self, self.trainer.datamodule)
                lr_eval = test_clf_lr_all_subsets(epoch, clf_lr, self, self.trainer.datamodule)
                for s, l_key in enumerate(sorted(lr_eval.keys())):
                    self.log('Latent Representation/%s' % l_key,
                             lr_eval[l_key], on_epoch=True)

            if self.config.evaluation['use_clf']:
                gen_eval = test_generation(self, self.trainer.datamodule)
                for j, l_key in enumerate(sorted(gen_eval['cond'].keys())):
                    for k, s_key in enumerate(gen_eval['cond'][l_key].keys()):
                        self.log('Generation/%s/%s' %
                                 (l_key, s_key),
                                 gen_eval['cond'][l_key][s_key],
                                 on_epoch=True)
                self.log('Generation/Random', gen_eval['random'], on_epoch=True)

            if self.config.evaluation['calc_nll']:
                lhoods = estimate_likelihoods(self, self.trainer.datamodule)
                for k, key in enumerate(sorted(lhoods.keys())):
                    self.log('Likelihoods/%s' % key,
                             lhoods[key], on_epoch=True)

            if self.config.evaluation['calc_prd'] and ((epoch + 1) % self.config.evaluation['eval_freq_fid'] == 0):
                prd_scores = calc_prd_score(self)
                self.log('PRD', prd_scores, on_epoch=True)

    def basic_routine(self, batch):
        beta_style = self.config.beta_values['beta_style']
        beta_content = self.config.beta_values['beta_content']
        beta = self.config.beta_values['beta']
        rec_weight = 1.0

        batch_d = batch[0]
        batch_l = batch[1]
        mods = self.modalities
        # not sure if needed
        for k, m_key in enumerate(batch_d.keys()):
            batch_d[m_key] = Variable(batch_d[m_key]).to(self.device)

        results = self(batch_d)

        log_probs, weighted_log_prob = self.calc_log_probs(results, batch)
        group_divergence = results['joint_divergence']

        klds = self.calc_klds(results)
        if self.config.method_mods['factorized_representation']:
            klds_style = self.calc_klds_style(results)

        if (self.config.method_mods['modality_jsd'] or self.config.method_mods['modality_moe'] or
                self.config.method_mods['joint_elbo']):
            if self.config.method_mods['factorized_representation']:
                kld_style = self.calc_style_kld(klds_style)
            else:
                kld_style = 0.0
            kld_content = group_divergence
            kld_weighted = beta_style * kld_style + beta_content * kld_content
            total_loss = rec_weight * weighted_log_prob + beta * kld_weighted
        elif self.config.method_mods['modality_poe']:
            klds_joint = {'content': group_divergence,
                          'style': dict()}
            elbos = dict()
            for m, m_key in enumerate(mods.keys()):
                mod = mods[m_key]
                if self.config.method_mods['factorized_representation']:
                    kld_style_m = klds_style[m_key + '_style']
                else:
                    kld_style_m = 0.0
                klds_joint['style'][m_key] = kld_style_m
                if self.config.method_mods['poe_unimodal_elbos']:
                    i_batch_mod = {m_key: batch_d[m_key]}
                    r_mod = self(i_batch_mod)
                    log_prob_mod = -mod.calc_log_prob(r_mod['rec'][m_key],
                                                      batch_d[m_key],
                                                      self.config.batch_size)
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

    def configure_optimizers(self):
        optimizer = optim.Adam(
            list(self.model.parameters()),
            lr=self.config.LR,
            betas=(self.config.beta_values['beta_1'],
                   self.config.beta_values['beta_2)']))
        return optimizer

    def set_rec_weights(self):
        rec_weights = dict()
        if self.config.dataset == 'SVHN_MNIST_text':
            ref_mod_d_size = self.modalities['svhn'].data_size.numel()
            for k, m_key in enumerate(self.modalities.keys()):
                mod = self.modalities[m_key]
                numel_mod = mod.data_size.numel()
                rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
        elif self.config.dataset == 'MMNIST':
            for k, m_key in enumerate(self.modalities.keys()):
                mod = self.modalities[m_key]
                numel_mod = mod.data_size.numel()
                rec_weights[mod.name] = 1.0
        else:
            raise NotImplementedError
        return rec_weights

    def set_style_weights(self):
        if self.config.dataset == 'SVHN_MNIST_text':
            weights = dict()
            weights['mnist'] = self.config.beta_values['beta_m1_style']
            weights['svhn'] = self.config.beta_values['beta_m2_style']
            weights['text'] = self.config.beta_values['beta_m3_style']
        elif self.config.dataset == 'MMNIST':
            weights = {"m%d" % m: self.config.beta_values['beta_style']
                       for m in range(self.num_modalities)}
        else:
            raise NotImplementedError
        return weights

    def get_test_samples(self, num_images=10):
        dm = self.trainer.datamodule
        dataset_test = dm.dataset_test
        n_test = len(dataset_test)
        samples = []
        for i in range(num_images):
            while True:
                if self.config.dataset == 'MMNIST':
                    ix = random.randint(0, n_test-1)
                    sample, target = dataset_test[ix]
                else:
                    sample, target = dataset_test.__getitem__(random.randint(0, n_test))
                if target == i:
                    for k, key in enumerate(sample):
                        sample[key] = sample[key].to(self)
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
        # ToDo swap model creation to change from config
        if self.config.dataset == 'SVHN_MNIST_text':
            model_clf_m1 = None
            model_clf_m2 = None
            model_clf_m3 = None
            if self.config.evaluation['use_clf']:
                model_clf_m1 = ClfImgMNIST()
                model_clf_m1.load_state_dict(torch.load(os.path.join(self.config.dir['dir_clf'],
                                                                     self.config.clf_save_m1)))
                # model_clf_m1 = model_clf_m1.to(self.device)

                model_clf_m2 = ClfImgSVHN()
                model_clf_m2.load_state_dict(torch.load(os.path.join(self.config.dir['dir_clf'],
                                                                     self.config.clf_save_m2)))
                # model_clf_m2 = model_clf_m2.to(self.device)

                model_clf_m3 = ClfText(self.config)
                model_clf_m3.load_state_dict(torch.load(os.path.join(self.config.dir['dir_clf'],
                                                                     self.config.clf_save_m3)))
                # model_clf_m3 = model_clf_m3.to(self.device)

            clfs = {'mnist': model_clf_m1,
                    'svhn': model_clf_m2,
                    'text': model_clf_m3}
        elif self.config.dataset == 'MMNIST':
            clfs = {"m%d" % m: None for m in range(self.num_modalities)}
            if self.config.use_clf:
                for m, fp in enumerate(self.config.dir['pretrained_clf_paths']):
                    model_clf = ClfImgCMNIST()
                    model_clf.load_state_dict(torch.load(fp))
                    model_clf.to(self)
                    clfs["m%d" % m] = model_clf
                for m, clf in clfs.items():
                    if clf is None:
                        raise ValueError("Classifier is 'None' for modality %s" % str(m))
        return clfs

    def set_modalities(self):
        #TODO swap such that simply modify from config
        if self.config.dataset == 'SVHN_MNIST_text':
            mod1 = MNIST('mnist', EncoderImg(self.config), DecoderImg(self.config),
                         self.config.class_dim, self.config.style_m1_dim, 'laplace')
            mod2 = SVHN('svhn', EncoderSVHN(self.config), DecoderSVHN(self.config),
                        self.config.class_dim, self.config.style_m2_dim, 'laplace',
                        self.plot_img_size)
            mod3 = Text('text', EncoderText(self.config), DecoderText(self.config),
                        self.config.class_dim, self.config.style_m3_dim, 'categorical',
                        self.config.len_sequence,
                        self.alphabet,
                        self.plot_img_size,
                        self.font)
            mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}
            return mods
        elif self.config.dataset == 'MMNIST':
            mods = [CMNIST("m%d" % m, EncImgCMNIST(self.config),
                           DecImgCMNIST(self.config), self.config.class_dim,
                           self.config.style_dim, self.config.likelihood) for m in
                    range(self.num_modalities)]
            mods_dict = {m.name: m for m in mods}
            return mods_dict
        else:
            raise NotImplementedError

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

    def set_paths_fid(self):
        dir_real = os.path.join(self.config.dir_gen_eval_fid, 'real')
        dir_random = os.path.join(self.config.dir_gen_eval_fid, 'random')
        paths = {'real': dir_real,
                 'random': dir_random}
        dir_cond = self.config.dir_gen_eval_fid
        for k, name in enumerate(self.subsets):
            paths[name] = os.path.join(dir_cond, name)
        print(paths.keys())
        return paths
