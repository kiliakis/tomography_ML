# Train the ML model
import matplotlib.pyplot as plt
from utils import sample_files, plot_loss, load_encoder_data
from utils import encoder_files_to_tensors, fast_tensor_load
from models import EncoderSingle
from itertools import product
import pickle
import time
import tensorflow as tf
import yaml
import os
import numpy as np
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')

import argparse



parser = argparse.ArgumentParser(description='Train the encoder/ decoder models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

# Initialize parameters
# data_dir = '/eos/user/k/kiliakis/tomo_data/datasets_encoder_02-12-22'
data_dir = './tomo_data/datasets_encoder_TF_24-03-23'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

DATA_LOAD_METHOD='FAST_TENSOR' # it can be TENSOR or DATASET or FAST_TENSOR
IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8
input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']
num_Turns_Case = 1

var_name = 'VrfSPS'

# Train specific
train_cfg = {
    'epochs': 25,
    'strides': [2, 2],
    'activation': 'relu',
    'pooling_size': [2, 2],
    'pooling_strides': [1, 1],
    'pooling_padding': 'valid',
    'dropout': 0.1,
    'loss': 'mse',
    'lr': 1e-3,
    'dataset%': 0.5,
    'normalization': 'minmax',
    'loss_weights': [var_names.index(var_name)],
    'img_normalize': 'off',
    'batch_size': 32
}

model_cfg = {
    var_name: {
        'cropping': [[0, 0], [7, 7]],
        'filters': [[8, 16, 32], [16, 32, 64]],
        'kernel_size': [[9, 7, 5], [7, 7, 7], [7, 5, 3], [5, 5, 5], [3, 3, 3]],
        'dense_layers': [[1024, 512, 128], [1024, 512, 64], [1024, 256, 64]],
        'strides': [[2, 2], [3, 3]],
        'pooling': [None, 'Max']
    },
}


def train_test_model(var_name, x_train, y_train, x_valid, y_valid, hparamdir, hparams):
    cfg = train_cfg.copy()
    cfg.update(model_cfg.get(var_name, {}))
    cfg.update(hparams)

    model = EncoderSingle(input_shape=input_shape,
                          output_name=var_name,
                          **cfg)

    start_t = time.time()
    history = model.model.fit(
        x=x_train, y=y_train,
        epochs=cfg['epochs'],
        validation_data=(x_valid, y_valid),
        callbacks=[],
        batch_size=cfg['batch_size'],
        verbose=0)
    total_t = time.time() - start_t
    val_loss = model.model.evaluate(x_valid, y_valid)

    # save file with experiment configuration
    # config_dict = {}
    # config_dict['encoder'] = cfg.copy()
    # config_dict['encoder'].update({
    #     'min_train_loss': float(np.min(history.history['loss'])),
    #     'min_valid_loss': float(np.min(history.history['val_loss'])),
    #     'total_train_time': total_t,
    #     'used_gpus': len(gpus)
    # })

    # save config_dict
#     with open(os.path.join(hparamdir, 'encoder-summary.yml'), 'w') as configfile:
#         yaml.dump(config_dict, configfile, default_flow_style=False)

    return history.history, val_loss


if __name__ == '__main__':
    args = parser.parse_args()

    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['encoder']
        if 'model_cfg' in input_config:
            model_cfg = input_config['model_cfg']
        timestamp = input_config['timestamp']
    
    print('Configuration:')
    for k, v in train_cfg.items():
        print(k, v)

    print('Model specific configuration:')
    for var in model_cfg:
        print(var, model_cfg[var])

    # Initialize directories
    # trial_dir = './'
    trial_dir = os.path.join('./hparam_trials/', timestamp)
    weights_dir = os.path.join(trial_dir, 'weights')
    plots_dir = os.path.join(trial_dir, 'plots')
    logs_dir = os.path.join(trial_dir, 'logs')
    hparams_dir = os.path.join(trial_dir, 'hparams')

    print('\n---- Using directory: ', trial_dir, ' ----\n')

    # Initialize GPU
    gpus = tf.config.experimental.list_physical_devices('GPU')
    device_to_use = 0

    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            # tf.config.experimental.set_memory_growth(gpus[device_to_use], True)
            # tf.config.experimental.set_virtual_device_configuration(
            #     gpus[device_to_use],
            #     [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12*1024)])
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(
                logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            print(e)
    else:
        print('No GPU available, using the CPU')

    # Initialize train/ test / validation paths
    ML_dir = os.path.join(data_dir, 'ML_data')
    assert os.path.exists(ML_dir)

    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')

    # create the directory to store the results
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(hparams_dir, exist_ok=True)

    np.random.seed(0)

    start_t = time.time()
    if DATA_LOAD_METHOD == 'TENSOR':
        # Create the datasets
        # 1. Randomly select the training data
        file_names = sample_files(
            TRAINING_PATH, train_cfg['dataset%'], keep_every=num_Turns_Case)
        print('Number of Training files: ', len(file_names))
        x_train, y_train = encoder_files_to_tensors(
            file_names, normalization=train_cfg['normalization'],
            img_normalize=train_cfg['img_normalize'])

        # Repeat for validation data
        file_names = sample_files(
            VALIDATION_PATH, train_cfg['dataset%'], keep_every=num_Turns_Case)
        print('Number of Validation files: ', len(file_names))

        x_valid, y_valid = encoder_files_to_tensors(
            file_names, normalization=train_cfg['normalization'],
            img_normalize=train_cfg['img_normalize'])
    elif DATA_LOAD_METHOD == 'FAST_TENSOR':
        assert train_cfg['normalization'] == 'minmax'
        assert train_cfg['img_normalize'] == 'off'

        TRAINING_PATH = os.path.join(ML_dir, 'training-??.npz')
        VALIDATION_PATH = os.path.join(ML_dir, 'validation-??.npz')

        x_train, y_train = fast_tensor_load(
            TRAINING_PATH, train_cfg['dataset%'])
        print('Number of Training files: ', len(y_train))

        x_valid, y_valid = fast_tensor_load(
            VALIDATION_PATH, train_cfg['dataset%'])
        print('Number of Validation files: ', len(y_valid))

    else:
        exit('DATA_LOAD_METHOD not recognised')

    print(
        f'\n---- Input files have been read, elapsed: {time.time() - start_t} ----\n')

    # Model instantiation
    start_t = time.time()
    input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)
    models = {}

    for i in train_cfg['loss_weights']:
        var_name = var_names[i]
        print(f'\n---- HyperParam tuning for: {var_name} ----\n')
        session_num = 0

        # Load data
        train = tf.gather(y_train, var_names.index(var_name), axis=1)
        valid = tf.gather(y_valid, var_names.index(var_name), axis=1)
        
        # Load param space
        param_space = model_cfg.get(var_name, {})
        keys, values = zip(*param_space.items())
        total_runs = np.prod([len(v) for v in param_space.values()])

        overall_dict = {}
        for bundle in product(*values):
            hparams = dict(zip(keys, bundle))
            run_name = f"run-{session_num}"
            print(f'--- [{var_name}] Starting trial: {run_name}/{total_runs}')
            print(hparams)
            start_t = time.time()
            history, loss = train_test_model(
                var_name, x_train, train, x_valid, valid, os.path.join(trial_dir, run_name), hparams)
            total_time = time.time() - start_t
            train_loss = np.min(history["loss"])
            valid_loss = np.min(history["val_loss"])
            overall_dict[run_name] = {
                'time': total_time, 'train': train_loss, 'valid': valid_loss, 'history': history, 'data%': train_cfg['dataset%']}
            overall_dict[run_name].update(hparams)
            print(
                f'---- {var_name} Training complete, epochs: {len(history["loss"])}, train loss {np.min(history["loss"]):.2e}, valid loss {np.min(history["val_loss"]):.2e}, time {total_time:.2f}')

            session_num += 1

            # save every 10 sessions
            if session_num % 10 == 0:
                fname = os.path.join(hparams_dir, f'{var_name}_{timestamp}.pkl')
                with open(fname, 'wb') as handle:
                    pickle.dump(overall_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)                

        # save final data
        fname = os.path.join(hparams_dir, f'{var_name}_{timestamp}.pkl')
        with open(fname, 'wb') as handle:
            pickle.dump(overall_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

