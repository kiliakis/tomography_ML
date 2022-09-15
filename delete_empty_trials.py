import os
import csv
import numpy as np
import argparse
import yaml
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='Delete trial directories if empty',
                                 usage='python ')

parser.add_argument('-i', '--indir', type=str, default='./trials',
                    help='The directory containing the trial data.')


if __name__ == '__main__':
    args = parser.parse_args()
    header = ['model', 'valid_loss', 'train_loss', 'cnn_filters',
              'epochs', 'dataset', 'lr', 'gpus', 'train_time', 'date']
    rows = []
    subdirs = os.listdir(args.indir)
    for subdir in subdirs:
        files = os.listdir(os.path.join(args.indir, subdir))
        if 'encoder-summary.yml' not in files and \
                'decoder-summary.yml' not in files:
            print(
                f'Directory {os.path.join(args.indir, subdir)} should be deleted')

