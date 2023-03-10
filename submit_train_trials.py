import os
import subprocess
from time import sleep
import yaml
from datetime import datetime
import argparse

submission_system = 'condor'
USERNAME = 'kiliakis'
RUNTIME = 8         # in hours
USE_GPU = 1          # request for a gpu node
CPU_CORES = 4        # number of CPU cores
if submission_system == 'condor':
    WORK = f"/afs/cern.ch/work/{USERNAME[0]}/{USERNAME}"
    # WORK = f"/eos/user/{USERNAME[0]}/{USERNAME}"
    PROJECT_DIR = f"{WORK}/git/tomography_ML"
    PYTHON = f'{WORK}/install/anaconda3/bin/python3'
else:
    print('Invalid submission system')
    exit()
ENCODER_SCRIPT = 'train_encoder_multiout.py'
DECODER_SCRIPT = 'train_decoder.py'

TRIALS_DIR = os.path.join(PROJECT_DIR, 'trials')

configs = [

    # {
    #     'encoder': {
    #         'epochs': 100,
    #         'strides': [2, 2],
    #         'activation': 'relu',
    #         'pooling': None, 'pooling_size': [2, 2],
    #         'pooling_strides': [1, 1], 'pooling_padding': 'valid',
    #         'dropout': 0.0,
    #         'loss': 'mse', 
    #         'lr': 1e-3,
    #         'dataset%': 1,
    #         'normalization': 'minmax',
    #         'img_normalize': 'off',
    #         'loss_weights': [1],
    #         'batch_size': 32
    #     },
    #     'model_cfg': {
    #         'phEr': {
    #             'cropping': [0, 0],
    #             'filters': [4, 8],
    #             'kernel_size': [3, 3],
    #             'dense_layers': [1024, 256, 32],
    #         },
    #         'enEr': {
    #             'cropping': [0, 0],
    #             'filters': [4, 8, 16],
    #             'kernel_size': [5, 5, 5],
    #             'dense_layers': [1024, 512, 64],
    #         },
    #         'bl': {
    #             'cropping': [0, 0],
    #             'filters': [8, 16, 32],
    #             'kernel_size': [5, 5, 5],
    #             'dense_layers': [1024, 512, 64],
    #         },
    #         'inten': {
    #             'cropping': [0, 0],
    #             'filters': [8, 16, 32],
    #             'kernel_size': [5, 5, 5],
    #             'dense_layers': [1024, 256, 64],
    #         },
    #         'Vrf': {
    #             'cropping': [6, 6],
    #             'filters': [8, 16, 32],
    #             'kernel_size': [5, 5, 5],
    #             'dense_layers': [1024, 256, 64],
    #         },
    #         'mu': {
    #             'cropping': [0, 0],
    #             'filters': [8, 16, 32],
    #             'kernel_size': [9, 9, 9],
    #             'dense_layers': [1024, 256, 128],
    #         },
    #         'VrfSPS': {
    #             'cropping': [0, 0],
    #             'filters': [4, 16, 32],
    #             'kernel_size': [7, 7, 7],
    #             'dense_layers': [1024, 256, 64],
    #         },
    #     }
    # },

    {
        'decoder': {
            'epochs': 100,
            'dense_layers': [8, 256, 1024],
            'filters': [32, 16, 8, 1],
            'kernel_size': 5,
            'activation': 'relu',
            'strides': [2, 2],
            'final_kernel_size': 5,
            'final_activation': 'tanh',
            'dropout': 0.0,
            'loss': 'mse',
            'lr': 1e-3,
            'dataset%': 0.6,
            'normalization': 'minmax',
            'ps_normalize': 'off',
            'loss_weights': [0, 1, 2, 3, 4, 5, 6, 7],
        },
    },
    {
        'decoder': {
            'epochs': 100,
            'dense_layers': [8, 256, 1024],
            'filters': [32, 16, 8, 1],
            'kernel_size': 7,
            'activation': 'relu',
            'strides': [2, 2],
            'final_kernel_size': 5,
            'final_activation': 'tanh',
            'dropout': 0.0,
            'loss': 'mse', 
            'lr': 1e-3,
            'dataset%': 0.6,
            'normalization': 'minmax',
            'ps_normalize': 'off',
            'loss_weights': [0, 1, 2, 3, 4, 5, 6, 7],

        },
    },
    {
        'decoder': {
            'epochs': 100,
            'dense_layers': [8, 256, 1024],
            'filters': [32, 16, 8, 1],
            'kernel_size': 9,
            'activation': 'relu',
            'strides': [2, 2],
            'final_kernel_size': 5,
            'final_activation': 'tanh',
            'dropout': 0.0,
            'loss': 'mse', 
            'lr': 1e-3,
            'dataset%': 0.6,
            'normalization': 'minmax',
            'ps_normalize': 'off',
            'loss_weights': [0, 1, 2, 3, 4, 5, 6, 7],
        },
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
        config_file_name = os.path.join(trial_dir, 'trial_input_config.yml')
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
