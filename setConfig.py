import argparse
import os
import ruamel.yaml

if __name__ == '__main__':
    argParser = argparse.ArgumentParser()
    argParser.add_argument('--config', help='your config file',
                           default='./configs/svhn_mnist_text.yaml')
    argParser.add_argument('--data_path', help='data path', default='../data')
    argParser.add_argument('--clf', help='clf path', default='../clf')
    argParser.add_argument('--inception_state_dict', help='path to inception V3 state dict',
                           default='../pt_inception-2015-12-05-6726825d.pth')
    argParser.add_argument('--fid', help='fid path', default=None)

    args = argParser.parse_args()

    yaml = ruamel.yaml.YAML()
    yaml.preserve_quotes = True
    with open(f'{args.config}', 'r') as f:
        config = yaml.load(f)

    config['dir']['data_path'] = os.path.expandvars(args.data_path)
    config['dir']['clf_path'] = os.path.expandvars(args.clf)
    config['dir']['fid_path'] = os.path.expandvars(args.fid)
    config['dir']['inception_path'] = os.path.expandvars(args.inception_state_dict)

    with open(f'{args.config}', 'w') as f:
        yaml.dump(config, f)
