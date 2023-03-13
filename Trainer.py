import pytorch_lightning as pl
from pytorch_lightning import Trainer, callbacks
from torch.utils.data import DataLoader

from eval_metrics.coherence import test_generation
from eval_metrics.likelihood import estimate_likelihoods
from eval_metrics.representation import train_clf_lr_all_subsets, test_clf_lr_all_subsets
from eval_metrics.sample_quality import calc_prd_score
from run_ import basic_routine_epoch


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
