import os
import csv
import numpy as np
import argparse
import yaml
from prettytable import PrettyTable
import glob

parser = argparse.ArgumentParser(description='Submit multiple train trials in htcondor',
                                 usage='python train_scan.py')

parser.add_argument('-i', '--indir', type=str, default='./trials/*',
                    help='The directory containing the collected data.')

parser.add_argument('-o', '--outfile', type=str, default='trials-sorted.csv',
                    help='The file to save the sorted report.'
                    ' Default: trials-sorted.csv')

parser.add_argument('-c', '--columns', nargs='+', type=int, default=[],
                    help='Which columns to show in stdout.'
                    ' Default: all columns')

parser.add_argument('-r', '--rows', type=int, default=-1,
                    help='How many rows to show in stdout.'
                    ' Default: all rows')

parser.add_argument('-u', '--update', action='store_true',
                    help='Update the csv file.'
                    ' Default: Do not update it')


def extract_trials(reg_expr_indir):
    header = ['model', 'vld_ls', 'trn_ls', 'epoch', 'data',
              'filters', 'kernels', 'dense', 'activ', 'dropout',
              'pooling', 'crop', 'lr', 'gpu', 'time', 'date']
    rows = []
    alldirs = glob.glob(reg_expr_indir)
    for indir in alldirs:
        if not os.path.isdir(indir):
            continue
        for dirs, subdirs, files in os.walk(indir):
            for file in files:
                if 'summary.yml' not in file:
                    continue
                with open(os.path.join(dirs, file)) as f:
                    summary = yaml.load(f, Loader=yaml.FullLoader)
                for k, v in summary.items():
                    model = k[:3]
                    date = os.path.basename(dirs)
                    kernels = f'{v.get("kernel_size", 3)}, {v.get("strides", 2)}'
                    pooling = f'{v.get("pooling", 0), v.get("pooling_size", 0), v.get("pooling_strides", 0), v.get("pooling_padding", 0)}'
                    filters = '-'.join(map(str, v.get('filters', [0])))
                    dense = '-'.join(map(str, v.get('dense_layers', [0])))
                    crop = '-'.join(map(str, v.get('cropping', [0, 0])))
                    activation = v.get('activation', 'linear')
                    if model == 'dec':
                        final_kernel_size = v.get('final_kernel_size', 3)
                        final_activation = v.get('final_activation', 'linear')
                        kernels = f'{v.get("kernel_size", 3)}-{final_kernel_size}, {v.get("strides", 2)}'
                        activation = f'{activation}-{final_activation}'
                    # date = date.replace('-', '').replace('_', '', 2)
                    row = [model, f'{v.get("min_valid_loss", 0):.2e}', f'{v.get("min_train_loss", 0):.2e}',
                           str(v.get('epochs', 0)), str(v.get('dataset%', 0)), filters,
                           kernels, dense, activation, str(v.get('dropout', 0)),
                           pooling, crop, str(v.get('lr', 0)), str(v.get('used_gpus', 0)),
                           f'{v.get("total_train_time", 0):.1f}', date]
                    rows.append(row)

    rows = sorted(rows, key=lambda a: (a[0], float(a[1]), float(a[2])))
    return header, rows


if __name__ == '__main__':
    args = parser.parse_args()

    header, rows = extract_trials(args.indir)

    # print the results
    if len(args.columns):
        idx = args.columns
        header = [header[i] for i in idx]
        # header = header[idx]
    else:
        idx = np.arange(len(header), dtype=int)

    encode_t = PrettyTable(header)
    encode_t.align = 'l'
    encode_t.border = False

    decode_t = PrettyTable(header)
    decode_t.align = 'l'
    decode_t.border = False

    for r in rows:
        if 'enc' in r[0]:
            r = [r[i] for i in idx]
            encode_t.add_row(r)
        if 'dec' in r[0]:
            r = [r[i] for i in idx]
            decode_t.add_row(r)

    if args.rows > 0:
        print(encode_t[:args.rows])
    else:
        print(encode_t)

    print()
    if args.rows > 0:
        print(decode_t[:args.rows])
    else:
        print(decode_t)

    if args.update:
        if os.path.isfile(args.outfile):
            # if file exists, I open and append to it
            with open(args.outfile, 'r') as f:
                reader = csv.reader(f, delimiter='\t')
                # make sure the header is the same
                file_header = next(reader)
                assert header == file_header
                # convert to set to remove duplicates
                rows = ['@'.join(row) for row in rows]
                rows = set(rows)
                for row in reader:
                    rows.add('@'.join(row))
            # split back and re-sort
            rows = [row.split('@') for row in rows]
            rows = sorted(rows, key=lambda a: (a[0], float(a[1]), float(a[2])))

        # Then save to it
        with open(args.outfile, 'w') as f:
            writer = csv.writer(f, delimiter='\t')
            writer.writerow(header)
            writer.writerows(rows)
