import pytorch_lightning as pl
from pytorch_lightning import Trainer, callbacks
from pytorch_lightning.callbacks import TQDMProgressBar
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from eval_metrics.coherence import test_generation
from eval_metrics.likelihood import estimate_likelihoods
from eval_metrics.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from eval_metrics.sample_quality import calc_prd_score
from plotting import generate_plots
from run_ import basic_routine_epoch

import sys
import os
import json

import torch

from run_ import run_epochs
from utils.TBlogger import TBLogger

from utils.filehandling import create_dir_structure
from utils.filehandling import create_dir_structure_testing
from MNISTSVHNTEXT.flags import parser
from MNISTSVHNTEXT.experiment import MNISTSVHNText
from MNISTSVHNTEXT.SVHNMNISTDataModule import SVHNMNISTDataModule

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    use_cuda = torch.cuda.is_available()
    FLAGS.device = torch.device('cuda' if use_cuda else 'cpu')

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
    # need to rewrite the experiment and run epoch part
    mst = MNISTSVHNText(FLAGS, alphabet)
    create_dir_structure_testing(mst)

    total_params = sum(p.numel() for p in mst.mm_vae.parameters())
    print('num parameters model: ' + str(total_params))

    dm = SVHNMNISTDataModule(FLAGS, alphabet)
    writer = SummaryWriter(mst.flags.dir_logs)
    tb_logger = TBLogger(mst.flags.str_experiment, writer)

    trainer = pl.Trainer(accelerator='gpu', devices=1, max_epochs=10, fast_dev_run=True, logger=tb_logger,
                         callbacks=[TQDMProgressBar(refresh_rate=20)])

    trainer.fit(mst, dm)

    result = trainer.test(mst, dm)

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
