import os
import csv
import numpy as np
import argparse
import yaml
from prettytable import PrettyTable

parser = argparse.ArgumentParser(description='Submit multiple train trials in htcondor',
                                 usage='python train_scan.py')

parser.add_argument('-i', '--indir', type=str, default='./trials',
                    help='The directory containing the collected data.')

parser.add_argument('-o', '--outfile', type=str, default='trials-sorted.csv',
                    help='The file to save the sorted report.'
                    ' Default: trials-sorted.csv')

if __name__ == '__main__':
    args = parser.parse_args()
    header = ['model', 'vld_ls', 'trn_ls', 'epoch', 'data',
              'filters', 'kernels', 'dense', 'activ', 'dropout', 
              'pooling', 'crop', 'lr', 'gpu', 'time', 'date']
    rows = []
    for dirs, subdirs, files in os.walk(args.indir):
        for file in files:
            if 'summary.yml' not in file:
                continue
            with open(os.path.join(dirs, file)) as f:
                summary = yaml.load(f, Loader=yaml.FullLoader)
            for k, v in summary.items():
                date = os.path.basename(dirs)
                kernels = f'{v["kernel_size"], v["strides"]}'
                pooling = f'{v["pooling"], v["pooling_size"], v["pooling_strides"], v["pooling_padding"]}'
                filters = '-'.join(map(str, v['filters']))
                dense = '-'.join(map(str, v['dense_layers']))
                crop = '-'.join(map(str, v['cropping']))

                # date = date.replace('-', '').replace('_', '', 2)
                row = [k[:3], f'{v["min_valid_loss"]:.4f}', f'{v["min_train_loss"]:.4f}',
                    str(v['epochs']), str(v['dataset%']), filters, kernels,
                     dense, v['activation'], str(v['dropout']), pooling, crop,
                    str(v['lr']), str(v['used_gpus']), f'{v["total_train_time"]:.1f}',
                    date]
                rows.append(row)
    
    rows = sorted(rows, key=lambda a: (a[0], a[1]))

    # print the results
    encode_t = PrettyTable(header)
    encode_t.align = 'l'
    encode_t.border = False
    
    decode_t = PrettyTable(header)
    decode_t.align = 'l'
    decode_t.border = False

    for r in rows:
        if 'enc' in r[0]:
            encode_t.add_row(r)
        if 'dec' in r[0]:
            decode_t.add_row(r)
    print(encode_t)
    print()
    print(decode_t)

    with open(args.outfile, 'w') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(header)
        writer.writerows(rows)
