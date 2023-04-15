import argparse
import json
import os
import sys

import pytorch_lightning as pl
import torch
import yaml
from PIL import ImageFont
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboardX import SummaryWriter

from Config import Config
from LitModExp import MultiModVAE
from LitModule import LitModule
from MNISTSVHNTEXT.SVHNMNISTDataModule import SVHNMNISTDataModule
from MNISTSVHNTEXT.flags import parser
from utils.filehandling import create_dir_structure

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

    # set configs here
    parser2 = argparse.ArgumentParser()
    parser2.add_argument('-c', '--cfg', help='specify config file', metavar='FILE')
    parser2.add_argument('--batch_size', type=int, default=None)
    parser2.add_argument('--seed', type=int, metavar='S', default=None)
    args = parser2.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    if config.data_params.method == 'poe':
        config.data_params.method_mods.modality_poe = True
    elif config.data_params.method == 'moe':
        config.data_params.method_mods.modality_moe = True
    elif config.data_params.method == 'jsd':
        config.data_params.method_mods.modality_jsd = True
    elif config.data_params.method == 'joint_elbos':
        config.data_params.method_mods.joint_elbo = True
    else:
        print('method implemented...exit')
        sys.exit()
    print(config.data_params.method_mods.modality_poe)
    print(config.data_params.method_mods.modality_moe)
    print(config.data_params.method_mods.modality_jsd)
    print(config.data_params.method_mods.joint_elbo)

    config.alpha_modalities = [config['exp_params']['div_weight_uniform_content'],
                               config['exp_params']['div_weight_m1_content'],
                               config['exp_params']['div_weight_m2_content'],
                               config['exp_params']['div_weight_m3_content']]

    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))

    plot_img_size = torch.Size((3, 28, 28))
    font = ImageFont.truetype(font=config['data_params']['font_file'], size=38)

    config.num_features = len(alphabet)
    FLAGS.num_features = len(alphabet)

    dm = SVHNMNISTDataModule(config, alphabet)
    mm_vae = MultiModVAE(config, FLAGS, font, plot_img_size)
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'])

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    trainer = pl.Trainer(devices='auto', accelerator='auto', max_epochs=2, fast_dev_run=True, logger=tb_logger,
                         callbacks=[TQDMProgressBar(refresh_rate=20)])

    trainer.fit(mm_vae, datamodule=dm)
    results = trainer.test(mm_vae, dm)

    print(results)
