# Train the ML model
from utils import sample_files, decoder_files_to_tensors
from utils import plot_loss, load_decoder_data
from models import Decoder
from itertools import product
import pickle
import time
import glob
import tensorflow as tf
# from tensorboard.plugins.hparams import api as hp
from tensorflow import keras
import yaml
import os
import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl

num_Turns_Case = 1
# Initialize parameters
# data_dir = '/eos/user/k/kiliakis/tomo_data/datasets_decoder_02-12-22'
data_dir = './tomo_data/datasets_decoder_02-12-22'

IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8
latent_dim = 7
additional_latent_dim = 1
input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

# Train specific
train_cfg = {
    'epochs': 20,
    'dense_layers': [latent_dim + additional_latent_dim, 64, 256],
    'filters': [256, 256, 128, 64, 32, 1],
    'kernel_size': 3,
    'strides': [2, 2],
    'final_kernel_size': 3,
    'activation': 'relu',
    'dropout': 0.2,
    'loss': 'mse',
    'lr': 1e-3,
    'dataset%': 0.1,
    'normalization': 'minmax',
    'batch_size': 32
}

param_space = {
    'dropout': [0.0],
    'kernel_size': [3, 5, 7],
    'filters': [
        #[32, 16, 8, 1],
                [64, 32, 16, 1], [128, 64, 32, 1]
                ],
    'dense_layers': [[8, 64, 1024], [8, 256, 1024],
                     [8, 64, 256, 1024], [8, 32, 256, 1024]]
}


def train_test_model(x_train, y_train, x_valid, y_valid, hparamdir, hparams):
    cfg = train_cfg.copy()
    cfg.update(hparams)

    model = Decoder(input_shape, **cfg)
#     weights_dir = os.path.join(hparamdir, 'weights')
    start_t = time.time()
    history = model.model.fit(
        x=x_train, y=y_train,
        validation_data=(x_valid, y_valid),
        batch_size=cfg['batch_size'],
        # x_train,
        # validation_data=x_valid,
        epochs=cfg['epochs'],
        verbose=0)
    total_t = time.time() - start_t
    val_loss = model.model.evaluate(x_valid, y_valid)

    # save file with experiment configuration
    config_dict = {}
    config_dict['decoder'] = cfg.copy()
    config_dict['decoder'].update({
        'min_train_loss': float(np.min(history.history['loss'])),
        'min_valid_loss': float(np.min(history.history['val_loss'])),
        'total_train_time': total_t,
        'used_gpus': len(gpus)
    })

    # save config_dict
#     with open(os.path.join(hparamdir, 'decoder-summary.yml'), 'w') as configfile:
#         yaml.dump(config_dict, configfile, default_flow_style=False)

    return history.history, val_loss


if __name__ == '__main__':

    # Initialize directories
    # trial_dir = os.path.join('./trials/', timestamp)
    trial_dir = './'
    # weights_dir = os.path.join(trial_dir, 'weights')
    # plots_dir = os.path.join(trial_dir, 'plots')
    # logs_dir = os.path.join(trial_dir, 'logs')

    # Initialize train/ test / validation paths
    ML_dir = os.path.join(data_dir, 'ML_data')
    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')
    assert os.path.exists(TRAINING_PATH)
    assert os.path.exists(VALIDATION_PATH)

    # create the directory to store the results
#     os.makedirs(trial_dir, exist_ok=True)
    # os.makedirs(weights_dir, exist_ok=False)
    # os.makedirs(plots_dir, exist_ok=False)
    # os.makedirs(logs_dir, exist_ok=False)

    # Initialize GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    device_to_use = 0

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            tf.config.experimental.set_memory_growth(gpus[device_to_use], True)
            tf.config.experimental.set_virtual_device_configuration(
                gpus[device_to_use],
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12*1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print('No GPU available, using the CPU')

    start_t = time.time()
    # Create the datasets
    # 1. Randomly select the training data
    file_names = sample_files(
        TRAINING_PATH, train_cfg['dataset%'], keep_every=num_Turns_Case)
    print('Number of Training files: ', len(file_names))

    x_train, y_train = decoder_files_to_tensors(
        file_names, normalization=train_cfg['normalization'])

    # Repeat for validation data
    file_names = sample_files(
        VALIDATION_PATH, train_cfg['dataset%'], keep_every=num_Turns_Case)
    print('Number of Validation files: ', len(file_names))

    x_valid, y_valid = decoder_files_to_tensors(
        file_names, normalization=train_cfg['normalization'])

    end_t = time.time()
    print(
        f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')

    session_num = 0
    keys, values = zip(*param_space.items())
    total_runs = np.prod([len(v) for v in param_space.values()])

    overall_dict = {}
    for bundle in product(*values):
        hparams = dict(zip(keys, bundle))
        run_name = f"run-{session_num}"
        print(f'--- Starting trial: {run_name}/{total_runs}')
        print(hparams)
        start_t = time.time()
        history, loss = train_test_model(x_train, y_train, x_valid, y_valid,
                                         os.path.join(trial_dir, run_name), hparams)
        total_time = time.time() - start_t
        train_loss = np.min(history["loss"])
        valid_loss = np.min(history["val_loss"])
        overall_dict[run_name] = {
            'time': total_time, 'train': train_loss, 'valid': valid_loss, 'history': history}
        overall_dict[run_name].update(hparams)
        print(
            f'---- Training complete, epochs: {len(history["loss"])}, train loss {np.min(history["loss"]):.2e}, valid loss {np.min(history["val_loss"]):.2e}, total time {total_time} ----')

        session_num += 1

    timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.makedirs('hparam_dicts', exist_ok=True)
    fname = f'hparam_dicts/decoder_{timestamp}.pkl'
    with open(fname, 'wb') as handle:
        pickle.dump(overall_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
