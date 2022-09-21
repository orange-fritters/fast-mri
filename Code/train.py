"""
Train File for the FastMRI Challenge
"""

import argparse
import os
import sys

from utils.learning.train_part import train
from pathlib import Path

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')


def parse():
    parser = argparse.ArgumentParser(description='Train Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU-NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-e', '--num-epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('-l', '--lr', type=float, default=3e-5, help='Learning rate')
    parser.add_argument('-r', '--report-interval', type=int, default=10, help='Report interval')
    parser.add_argument('-t', '--data-path-train', type=Path, default='../input/train',
                        help='Directory of train data')
    parser.add_argument('-v', '--data-path-val', type=Path, default='../input/val',
                        help='Directory of validation data')

    parser.add_argument('--cascade', type=int, default=12, help='Number of cascades | Should be less than 12')
    parser.add_argument('--input-key', type=str, default='kspace', help='Name of input key')
    parser.add_argument('--target-key', type=str, default='image_label', help='Name of target key')
    parser.add_argument('--max-key', type=str, default='max', help='Name of max key in attributes')

    parser.add_argument('--n_feats', type=int, default=160, help='Number of feature maps')
    parser.add_argument('--n_resblocks', type=int, default=25, help='Number of ResBlocks')
    parser.add_argument('--n_resgroups', type=int, default=4, help='Number of ResGroups')
    parser.add_argument('--reduction', type=int, default=16, help='Reduction')
    parser.add_argument('--res_scale', type=float, default=0.125, help='Res Mul Scale')
    parser.add_argument('--initialization', type=float, default=0.3, help='Initialization')

    parser.add_argument('-n', '--net-name', type=Path, default='TEST', help='Name of network')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.val_dir = '../result' / args.net_name / 'reconstructions_val'
    args.main_dir = '../result'/ args.net_name / __file__

    args.exp_dir.mkdir(parents=True, exist_ok=True)
    args.val_dir.mkdir(parents=True, exist_ok=True)

    train(args)
