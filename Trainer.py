import random
from itertools import chain, combinations

import PIL.Image
import numpy as np
import pytorch_lightning as pl
from PIL import ImageFont
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks import TQDMProgressBar
from sklearn.metrics import accuracy_score
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms as transforms

from MNISTSVHNTEXT.ConvNetImgMNIST import EncoderImg, DecoderImg
from eval_metrics.coherence import test_generation
from eval_metrics.likelihood import estimate_likelihoods
from eval_metrics.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from eval_metrics.sample_quality import calc_prd_score
from modalities.MNIST import MNIST
from plotting import generate_plots
from run_ import basic_routine_epoch

import sys
import os
import json

import torch

from run_ import run_epochs
from utils.TBlogger import TBLogger
from pytorch_lightning.loggers import TensorBoardLogger
from utils.filehandling import create_dir_structure
from utils.filehandling import create_dir_structure_testing
from MNISTSVHNTEXT.flags import parser
from LitModule import LitModule
from MNISTSVHNTEXT.experiment import MNISTSVHNText
from MNISTSVHNTEXT.SVHNMNISTDataModule import SVHNMNISTDataModule
from modalities.MNIST import MNIST
from modalities.SVHN import SVHN
from modalities.Text import Text

from MNISTSVHNTEXT.SVHNMNISTDataset import SVHNMNIST
from MNISTSVHNTEXT.VAEtrimodalSVHNMNIST import VAEtrimodalSVHNMNIST
from MNISTSVHNTEXT.ConvNetImgClfMNIST import ClfImg as ClfImgMNIST
from MNISTSVHNTEXT.ConvNetImgClfSVHN import ClfImgSVHN
from MNISTSVHNTEXT.ConvNetTextClf import ClfText as ClfText

from MNISTSVHNTEXT.ConvNetImgMNIST import EncoderImg, DecoderImg
from MNISTSVHNTEXT.ConvNetImgSVHN import EncoderSVHN, DecoderSVHN
from MNISTSVHNTEXT.ConvNetTextMNIST import EncoderText, DecoderText


def set_modalities():
    mod1 = MNIST('mnist', EncoderImg(FLAGS), DecoderImg(FLAGS),
                 FLAGS.class_dim, FLAGS.style_m1_dim, 'laplace')
    mod2 = SVHN('svhn', EncoderSVHN(FLAGS), DecoderSVHN(FLAGS),
                FLAGS.class_dim, FLAGS.style_m2_dim, 'laplace',
                plot_img_size)
    mod3 = Text('text', EncoderText(FLAGS), DecoderText(FLAGS),
                FLAGS.class_dim, FLAGS.style_m3_dim, 'categorical',
                FLAGS.len_sequence,
                alphabet,
                plot_img_size,
                font)
    mods = {mod1.name: mod1, mod2.name: mod2, mod3.name: mod3}
    return mods


def set_subsets():
    num_mods = len(list(modalities.keys()))

    """
        powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3)
        (1,2,3)
        """
    xs = list(modalities)
    # note we return an iterator rather than a list
    subsets_list = chain.from_iterable(combinations(xs, n) for n in
                                       range(len(xs) + 1))
    subsets = dict()
    for k, mod_names in enumerate(subsets_list):
        mods = []
        for l, mod_name in enumerate(sorted(mod_names)):
            mods.append(modalities[mod_name])
        key = '_'.join(sorted(mod_names))
        subsets[key] = mods
    return subsets


def set_clfs():
    model_clf_m1 = None
    model_clf_m2 = None
    model_clf_m3 = None
    if FLAGS.use_clf:
        model_clf_m1 = ClfImgMNIST()
        model_clf_m1.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf,
                                                             FLAGS.clf_save_m1)))

        model_clf_m2 = ClfImgSVHN()
        model_clf_m2.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf,
                                                             FLAGS.clf_save_m2)))

        model_clf_m3 = ClfText(FLAGS)
        model_clf_m3.load_state_dict(torch.load(os.path.join(FLAGS.dir_clf,
                                                             FLAGS.clf_save_m3)))

    clfs = {'mnist': model_clf_m1,
            'svhn': model_clf_m2,
            'text': model_clf_m3}
    return clfs


def set_rec_weights():
    rec_weights = dict()
    ref_mod_d_size = modalities['svhn'].data_size.numel()
    for k, m_key in enumerate(modalities.keys()):
        mod = modalities[m_key]
        numel_mod = mod.data_size.numel()
        rec_weights[mod.name] = float(ref_mod_d_size / numel_mod)
    return rec_weights


def set_style_weights():
    weights = dict()
    weights['mnist'] = FLAGS.beta_m1_style
    weights['svhn'] = FLAGS.beta_m2_style
    weights['text'] = FLAGS.beta_m3_style
    return weights


def mean_eval_metric(self, values):
    return np.mean(np.array(values))


def get_prediction_from_attr(self, attr, index=None):
    pred = np.argmax(attr, axis=1).astype(int)
    return pred


def eval_label(self, values, labels, index):
    pred = get_prediction_from_attr(values)
    return self.eval_metric(labels, pred)


def set_paths_fid():
    dir_real = os.path.join(FLAGS.dir_gen_eval_fid, 'real')
    dir_random = os.path.join(FLAGS.dir_gen_eval_fid, 'random')
    paths = {'real': dir_real,
             'random': dir_random}
    dir_cond = FLAGS.dir_gen_eval_fid
    for k, name in enumerate(subsets):
        paths[name] = os.path.join(dir_cond, name)
    print(paths.keys())
    return paths


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    # use_cuda = torch.cuda.is_available()
    # FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')

    if FLAGS.method == 'poe':
        FLAGS.modality_poe = True
    elif FLAGS.method == 'moe':
        FLAGS.modality_moe = True
    elif FLAGS.method == 'jsd':
        FLAGS.modality_jsd = True
    elif FLAGS.method == 'joint_elbo':
        FLAGS.joint_elbo = True
    else:
        print('method implemented...exit!')
        sys.exit()
    print(FLAGS.modality_poe)
    print(FLAGS.modality_moe)
    print(FLAGS.modality_jsd)
    print(FLAGS.joint_elbo)

    FLAGS.alpha_modalities = [FLAGS.div_weight_uniform_content, FLAGS.div_weight_m1_content,
                              FLAGS.div_weight_m2_content, FLAGS.div_weight_m3_content]

    FLAGS = create_dir_structure(FLAGS)
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))
    # need to rewrite the experiment setup and run epoch part (mainly basic_epoch_routine)
    # basic epoch routine left to implement
    plot_img_size = torch.Size((3, 28, 28))
    font = ImageFont.truetype('FreeSerif.ttf', 38)  # seems to cause problems
    FLAGS.num_features = len(alphabet)
    modalities = set_modalities()
    num_modalities = len(modalities.keys())
    subsets = set_subsets()
    mm_vae = LitModule(FLAGS, modalities, subsets)
    clfs = set_clfs()
    rec_weights = set_rec_weights()
    style_weights = set_style_weights()

    # test_samples = get_test_samples() not sure what to do here
    eval_metrics = accuracy_score
    paths_fic = set_paths_fid()

    labels = ['digit']

    create_dir_structure_testing(mm_vae.flags, labels)

    total_params = sum(p.numel() for p in mm_vae.parameters())
    print('num parameters model: ' + str(total_params))
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.ToPILImage(),
                                          transforms.Resize(size=(28, 28), interpolation=PIL.Image.BICUBIC),
                                          transforms.ToTensor()])
    transform_svhn = transforms.Compose([transforms.ToTensor()])
    transform = [transform_mnist, transform_svhn]
    train_set = SVHNMNIST(FLAGS, alphabet, train=True, transform=transform)
    train = DataLoader(train_set, batch_size=FLAGS.batch_size,
                       shuffle=True, num_workers=8, drop_last=True)
    dm = SVHNMNISTDataModule(FLAGS, alphabet)
    writer = SummaryWriter(mm_vae.flags.dir_logs)
    tb_logger = TBLogger(mm_vae.flags.str_experiment, writer)
    logger2 = TensorBoardLogger("tb_logs", name="Lit_Model")

    trainer = Trainer(devices=1, max_epochs=10, fast_dev_run=True, logger=logger2,
                      callbacks=[TQDMProgressBar(refresh_rate=20)])

    trainer.fit(mm_vae, train_dataloaders=train)

    result = trainer.test(mm_vae, dm)

    print(result)


class TriModTrainer(callbacks.Callback):
    def __init__(self):
        pass

    def _train(self, exp, tb_logger):
        mm_vae = exp.mm_vae
        mm_vae.train()
        exp.mm_vae = mm_vae

        d_loader = DataLoader(exp.dataset_train, batch_size=exp.flags.batch_size,
                              shuffle=True,
                              num_workers=8, drop_last=True)

        for iteration, batch in enumerate(d_loader):
            basic_routine = basic_routine_epoch(exp, batch)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            # backprop
            exp.optimizer.zero_grad()
            total_loss.backward()
            exp.optimizer.step()
            tb_logger.write_training_logs(results, total_loss, log_probs, klds)

    def _test(self, exp, tb_logger, epoch):
        mm_vae = exp.mm_vae
        mm_vae.eval()
        exp.mm_vae = mm_vae

        # set up weights
        beta_style = exp.flags.beta_style
        beta_content = exp.flags.beta_content
        beta = exp.flags.beta
        rec_weight = 1.0

        d_loader = DataLoader(exp.dataset_test, batch_size=exp.flags.batch_size,
                              shuffle=True,
                              num_workers=8, drop_last=True)

        for iteration, batch in enumerate(d_loader):
            basic_routine = basic_routine_epoch(exp, batch)
            results = basic_routine['results']
            total_loss = basic_routine['total_loss']
            klds = basic_routine['klds']
            log_probs = basic_routine['log_probs']
            tb_logger.write_testing_logs(results, total_loss, log_probs, klds)

        plots = generate_plots(exp, epoch)
        tb_logger.write_plots(plots, epoch)

        if (epoch + 1) % exp.flags.eval_freq == 0 or (epoch + 1) == exp.flags.end_epoch:
            if exp.flags.eval_lr:
                clf_lr = train_clf_lr_all_subsets(exp)
                lr_eval = test_clf_lr_all_subsets(epoch, clf_lr, exp)
                tb_logger.write_lr_eval(lr_eval)

            if exp.flags.use_clf:
                gen_eval = test_generation(epoch, exp)
                tb_logger.write_coherence_logs(gen_eval)

            if exp.flags.calc_nll:
                lhoods = estimate_likelihoods(exp)
                tb_logger.write_lhood_logs(lhoods)

            if exp.flags.calc_prd and ((epoch + 1) % exp.flags.eval_freq_fid == 0):
                prd_scores = calc_prd_score(exp)
                tb_logger.write_prd_scores(prd_scores)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pass

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pass

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        pass
