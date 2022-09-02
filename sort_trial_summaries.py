import os
import csv
import sys
import fnmatch
import numpy as np
import subprocess
import argparse
import glob
import yaml

parser = argparse.ArgumentParser(description='Submit multiple train trials in htcondor',
                                 usage='python train_scan.py')

parser.add_argument('-i', '--indir', type=str, default='./trials',
                    help='The directory containing the collected data.')

parser.add_argument('-o', '--outfile', type=str, default='trials-sorted.csv',
                    help='The file to save the sorted report.'
                    ' Default: trials-sorted.csv')

if __name__ == '__main__':
    args = parser.parse_args()
    header = ['model', 'valid_loss', 'train_loss', 'cnn_filters',
              'epochs', 'dataset', 'lr', 'gpus', 'train_time', 'date']
    rows = []
    for dirs, subdirs, files in os.walk(args.indir):
        if 'summary.yml' not in files:
            continue
        with open(os.path.join(dirs, 'summary.yml')) as f:
            summary = yaml.load(f, Loader=yaml.FullLoader)
        for k, v in summary.items():
            row = [k, f'{v["min_valid_loss"]:4f}', f'{v["min_train_loss"]:4f}',
                   '-'.join(map(str, v['cnn_filters'])), str(v['epochs']),
                   str(v['dataset_percent']), str(v['lr']),
                   str(v['used_gpus']), f'{v["total_train_time"]:1f}',
                   '0']
            rows.append(row)
    
    rows = sorted(rows, key=lambda a: (a[0], a[1]))
    with open(args.outfile, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)
