import argparse
import sys, os

import pytorch_lightning as pl
import yaml
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from tensorboardX import SummaryWriter
from LitModule import LitModule
from MNISTSVHNTEXT.SVHNMNISTDataModule import SVHNMNISTDataModule
from MNISTSVHNTEXT.flags import parser
from utils.TBlogger import TBLogger

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

    # set configs here

    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    mm_vae = LitModule()
    dm = SVHNMNISTDataModule(FLAGS, alphabet)
    # writer = setWriter()
    tb_logger = TensorBoardLogger(save_dir=config['logging_params']['save_dir'],
                                  name=config['model_params']['name'], )

    # For reproducibility
    seed_everything(config['exp_params']['manual_seed'], True)

    model = vae_models[config['model_params']['name']](**config['model_params'])

    trainer = pl.Trainer(devices=1, max_epochs=2, fast_dev_run=True, logger=tb_logger,
                         callbacks=[TQDMProgressBar(refresh_rate=20)])

    trainer.fit(mm_vae, dm)
    results = trainer.test(mm_vae, dm)

    print(results)

# def setWriter():
# return SummaryWriter(logdir=path)
