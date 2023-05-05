import argparse
import json
import os
import sys

import pytorch_lightning as pl
import torch
import yaml
import ruamel.yaml
from PIL import ImageFont
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from Config import Config
from LitModExp import MultiModVAE
from LitModule import LitModule
from MNISTSVHNTEXT.SVHNMNISTDataModule import SVHNMNISTDataModule
from MNISTSVHNTEXT.SVHNMNISTDataModuleConf import SVHNMNISTDataModuleC
from MMNIST.MMNISTDataModule import MMNISTDataModule
from MNISTSVHNTEXT.flags import parser
from utils.filehandling import create_dir_structure

if __name__ == '__main__':
    parser2 = argparse.ArgumentParser()
    use_cuda = torch.cuda.is_available()

    parser2.add_argument('--config', '-c',
                         dest='filename',
                         metavar='FILE',
                         help='path to config file',
                         default='configs/mmnist.yaml')
    config = Config(parser2)
    config.device = torch.device('cuda' if use_cuda else 'cpu')
    
    print('config.mods: ' + str(config.mods))
    print('config.method_mods: '+ str(config.method_mods))

    if config.method == 'poe':
        config.method_mods['modality_poe'] = True
    elif config.method == 'moe':
        config.method_mods['modality_moe'] = True
    elif config.method == 'jsd':
        config.method_mods['modality_jsd'] = True
    elif config.method == 'joint_elbos':
        config.method_mods['joint_elbo'] = True
    else:
        print('method implemented...exit')
        sys.exit()

    print(config.method_mods['modality_poe'])
    print(config.method_mods['modality_moe'])
    print(config.method_mods['modality_jsd'])
    print(config.method_mods['joint_elbo'])
    
    # set alpha_modalities and div_weights
    if hasattr(config, 'unimodal_datapaths'):
        assert len(config.unimodal_datapaths['train']) == len(config.unimodal_datapaths['test'])
        config.num_mods = len(config.unimodal_datapaths['train'])
    if config.div_weight['div_weight_uniform_content'] is None:
        config.div_weight['div_weight_uniform_content'] = 1 / (config.num_mods + 1)
    config.alpha_modalities = [config.div_weight['div_weight_uniform_content']]
    if 'div_weight' in config.div_weight:
        if config.div_weight['div_weight'] is None:
            config.div_weight['div_weight'] = 1 / (config.num_mods + 1)
        config.alpha_modalities.extend([config.div_weight['div_weight'] for _ in range(config.num_mods)])
    else:
        config.num_mods = len(config.mods)
        config.alpha_modalities.extend([config.mods[_]['div_weight'] for _ in range(config.num_mods)])

    create_dir_structure(config)
    alphabet_path = os.path.join(os.getcwd(), 'alphabet.json')
    with open(alphabet_path) as alphabet_file:
        alphabet = str(''.join(json.load(alphabet_file)))

    plot_img_size = torch.Size((3, 28, 28))
    font = ImageFont.truetype(font=config.font_file, size=38)
    
    print (config.alpha_modalities)
    print ('config.num_mods: ' + str(config.num_mods))

    # set correct LightningDataModule
    if config.dataset == 'MMNIST':
        dm = MMNISTDataModule(config, alphabet)
    else:
        dm = SVHNMNISTDataModuleC(config, alphabet)
    
    # init LightningModule and logger
    mm_vae = MultiModVAE(config, font, plot_img_size, alphabet)
    tb_logger = TensorBoardLogger(save_dir=config.logging_params['save_dir'],
                                  name=config.logging_params['name'])

    # For reproducibility
    seed_everything(config.manual_seed, True)
    # print(config.unimodal_datapaths['train'])

    # init Trainer (many more flags available)
    trainer = pl.Trainer(devices='auto', accelerator='gpu',
                         max_epochs=config.trainer_params['max_epochs'],
                         fast_dev_run=False, logger=tb_logger,
                         callbacks=[TQDMProgressBar(refresh_rate=20)])

    # main work part
    trainer.fit(mm_vae, datamodule=dm)
    # trainer.validate(mm_vae, dm)
    results = trainer.test(mm_vae, dm)

    print(results)
