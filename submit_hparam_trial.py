import os
import subprocess
from time import sleep
import yaml
from datetime import datetime
import argparse

submission_system = 'condor'
USERNAME = 'kiliakis'
RUNTIME = 5         # in hours
USE_GPU = 1          # request for a gpu node
CPU_CORES = 1        # number of CPU cores
if submission_system == 'condor':
    WORK = f"/afs/cern.ch/work/{USERNAME[0]}/{USERNAME}"
    # WORK = f"/eos/user/{USERNAME[0]}/{USERNAME}"
    PROJECT_DIR = f"{WORK}/git/tomography_ML"
    PYTHON = f'{WORK}/install/anaconda3/bin/python3'
else:
    print('Invalid submission system')
    exit()
ENCODER_SCRIPT = 'train_encoder_hyperparam_multiout.py'
DECODER_SCRIPT = 'train_decoder-hyperparam.py'

TRIALS_DIR = os.path.join(PROJECT_DIR, 'hparam_trials')

var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']

configs = [
    {
        'encoder': {
            'epochs': 25,
            'dense_layers': [1024, 512, 128],
            'filters': [8, 16, 32],
            'cropping': [0, 0],
            'kernel_size': [5, 5, 5],
            'activation': 'relu',
            'pooling_size': [2, 2],
            'pooling_strides': [1, 1],
            'pooling_padding': 'valid',
            'dropout': 0.1,
            'loss': 'mse',
            'lr': 1e-3,
            'dataset%': 0.75,
            'normalization': 'minmax',
            'img_normalize': 'off',
            'loss_weights': [5],
            'batch_size': 32
        },
        'model_cfg': {
            'mu': {
                'cropping': [[0, 0]],
                'kernel_size': [[13, 9, 7], [13, 9, 5],
                                [9, 9, 9], [9, 7, 5],
                                [7, 7, 7], [7, 5, 5],
                                ],
                'filters': [[4, 16, 32], [4, 16, 64],
                            [8, 16, 32], [8, 32, 64]],
                'dense_layers': [[1024,256,128], [768, 256, 128], [768, 256, 64],
                                 ],
                'strides': [[2, 2]],
                'pooling': ['Max']
            },
        }
    },

    {
        'encoder': {
            'epochs': 25,
            'dense_layers': [1024, 512, 128],
            'filters': [8, 16, 32],
            'cropping': [0, 0],
            'kernel_size': [5, 5, 5],
            'activation': 'relu',
            'pooling_size': [2, 2],
            'pooling_strides': [1, 1],
            'pooling_padding': 'valid',
            'dropout': 0.1,
            'loss': 'mse',
            'lr': 1e-3,
            'dataset%': 0.75,
            'normalization': 'minmax',
            'img_normalize': 'off',
            'loss_weights': [6],
            'batch_size': 32
        },
        'model_cfg': {
            'VrfSPS': {
                'cropping': [[0, 0]],
                'kernel_size': [[13, 9, 7], [13, 9, 5],
                                [9, 9, 9], [9, 7, 5],
                                [7, 7, 7], [7, 5, 5],
                                ],
                'filters': [[8, 16, 32], [8, 32, 64],
                            [16, 32, 32], [16, 32, 64]],
                'dense_layers': [[1024, 512, 256], [1024, 256, 128],
                                 [768, 512, 128], [512, 256, 128], [512, 256, 64],
                                 ],
                'strides': [[2, 2]],
                'pooling': ['Max']
            },
        }
    },

]

parser = argparse.ArgumentParser(description='Submit multiple train trials in htcondor',
                                 usage='python train_scan.py')

parser.add_argument('-dry', '--dry-run', action='store_true',
                    help='Do not submit, just prepare everything.')

if __name__ == '__main__':
    args = parser.parse_args()
    for config in configs:
        config['timestamp'] = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
        trial_dir = os.path.join(TRIALS_DIR, config['timestamp'])
        os.makedirs(trial_dir, exist_ok=False)
        # Change directory
        os.chdir(trial_dir)
        # Save config
        config_file_name = os.path.join(trial_dir, 'hparam_input_config.yml')
        with open(config_file_name, 'w') as file:
            yaml.dump(config, file, default_flow_style=False)

        # Create submission script
        SUBMISSION_SCRIPT = os.path.join(trial_dir, "submit.sub")
        print(f"Creating submission script {SUBMISSION_SCRIPT}")
        with open(SUBMISSION_SCRIPT, 'w') as f:
            f.write("executable              = execute.sh\n")
            f.write("arguments               = $(ClusterId)$(ProcId)\n")
            f.write(f"output                 = {trial_dir}/output.txt\n")
            f.write(f"error                  = {trial_dir}/error.txt\n")
            f.write(f"log                    = {trial_dir}/log.txt\n")
            f.write("getenv                  = True \n")
            f.write("should_transfer_files   = IF_needed \n")
            f.write(f"request_gpus            = {USE_GPU} \n")
            f.write(f"request_cpus            = {CPU_CORES} \n")
            f.write(f"request_memory            = 28000MB \n")
            # f.write(f"+RequestMemory            = 10000 \n")
            # f.write("requirements            = regexp(\"V100\", TARGET.CUDADeviceName) \n")
            # f.write("Arch                    = \"INTEL\" \n ")
            f.write(f"+MaxRuntime            = {int(3600 * RUNTIME)} \n")
            f.write("queue")

        # Create shell script
        SHELL_SCRIPT = os.path.join(trial_dir, "execute.sh")
        print(f"Creating shell script {SHELL_SCRIPT}")

        with open(SHELL_SCRIPT, 'w') as f:
            f.write(f"#!/bin/bash\n\n")
            f.write(f"# Make sure right paths are used\n")
            f.write(f'source $HOME/.bashrc\n')
            f.write(f"{PYTHON} --version\n")
            f.write("lscpu \n")
            f.write(f"cd {PROJECT_DIR}\n")
            f.write(f"export TF_GPU_ALLOCATOR=cuda_malloc_async\n")
            f.write(f"# Run the python script\n")
            if 'encoder' in config:
                f.write(
                    f"{PYTHON} {ENCODER_SCRIPT} -c {config_file_name}\n")
            if 'decoder' in config:
                f.write(
                    f"{PYTHON} {DECODER_SCRIPT} -c {config_file_name}\n")

        # Print the shell script content on the screen
        # subprocess.run(["cat", "execute.sh"])

        # sleep to avoid directory collisions
        sleep(1.5)
        if args.dry_run:
            continue

        # Submit the job to condor
        print("Submitting job")
        if submission_system == 'condor':
            subprocess.run(["condor_submit", "submit.sub"])

    print('Done!')