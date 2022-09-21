import argparse
import os
import sys
from pathlib import Path

from utils.learning.test_part import forward

if os.getcwd() + '/utils/model/' not in sys.path:
    sys.path.insert(1, os.getcwd() + '/utils/model/')


def parse():
    parser = argparse.ArgumentParser(description='Test Unet on FastMRI challenge Images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--GPU_NUM', type=int, default=0, help='GPU number to allocate')
    parser.add_argument('-b', '--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('-p', '--data_path', type=Path, default='../leaderboard', help='Directory of test data')

    parser.add_argument('--cascade', type=int, default=12, help='Number of cascades')
    parser.add_argument("--input_key", type=str, default='kspace', help='Name of input key')

    parser.add_argument('--n_feats', type=int, default=160, help='Number of feature maps')
    parser.add_argument('--n_resblocks', type=int, default=25, help='Number of ResBlocks')
    parser.add_argument('--n_resgroups', type=int, default=4, help='Number of ResGroups')
    parser.add_argument('--reduction', type=int, default=16, help='Reduction')
    parser.add_argument('--res_scale', type=float, default=0.25, help='Res Mul Scale')
    parser.add_argument('--initialization', type=float, default=0.15, help='Initialization')

    parser.add_argument('-n', '--net_name', type=Path, default='RCAN', help='Name of network')
    parser.add_argument('--fresh_trained', type=bool, default=False, help='Newly Trained or Not')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse()
    args.exp_dir = '../result' / args.net_name / 'checkpoints'
    args.forward_dir = '../result' / args.net_name / 'reconstructions_forward'
    forward(args)

