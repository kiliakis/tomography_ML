import os
import sys
import subprocess
from time import sleep
import yaml
from datetime import datetime
import argparse

submission_system = 'condor'
USERNAME = 'kiliakis'

if submission_system == 'condor':
    WORK = f"/afs/cern.ch/work/{USERNAME[0]}/{USERNAME}"
    # WORK = f"/eos/user/{USERNAME[0]}/{USERNAME}"
    PROJECT_DIR = f"{WORK}/git/tomography_ML"
else:
    print('Invalid submission system')
    exit()

PYTHON = sys.executable
ENCODER_SCRIPT = 'train_encoder_multiout.py'
DECODER_SCRIPT = 'train_decoder.py'
TOMOSCOPE_SCRIPT = 'train_tomoscope.py'

TRIALS_DIR = os.path.join(PROJECT_DIR, 'trials')

parser = argparse.ArgumentParser(description='Submit multiple train trials in htcondor',
                                 usage='python train_scan.py')

parser.add_argument('-dry', '--dry-run', action='store_true',
                    help='Do not submit, just prepare everything.')

parser.add_argument('-c', '--configs', nargs='+', type=str,
                    help='YAML files with trial configurations to run.')

parser.add_argument('-no-gpu', '--no-gpu', action='strore_true',
                    help='Do not request for a GPU node.')

parser.add_argument('-cores', '--cores', type=int, default=1,
                    help='Number of CPU cores to ask for.')

parser.add_argument('-time', '--time', type=int, default=1,
                    help='Runtime per cofiguration in hours.')

if __name__ == '__main__':
    args = parser.parse_args()
    RUNTIME = args.time
    USE_GPU = not args.no_gpu
    CPU_CORES = args.cores
    configs = []
    for yamlfile in args.configs:
        print(f'Loading {yamlfile}')
        with open(yamlfile) as f:
            temp_configs = yaml.load(f, Loader=yaml.FullLoader)
        print(f'{yamlfile} loaded, found {len(temp_configs)} configurations')

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
            f.write(f'requirements = ( TARGET.OpSysAndVer =?= "AlmaLinux9" || TARGET.OpSysAndVer =?= "CentOS7")\n')
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
            elif 'decoder' in config:
                f.write(
                    f"{PYTHON} {DECODER_SCRIPT} -c {config_file_name}\n")
            elif 'tomoscope' in config:
                f.write(
                    f"{PYTHON} {TOMOSCOPE_SCRIPT} -c {config_file_name}\n")


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
