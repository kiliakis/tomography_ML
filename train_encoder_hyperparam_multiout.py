# Train the ML model
from utils import sample_files, plot_loss, load_encoder_data
from utils import encoder_files_to_tensors
from models import EncoderSingle
from itertools import product
import pickle
import time
import tensorflow as tf
import yaml
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl

num_Turns_Case = 1
var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']
# Initialize parameters
# data_dir = '/eos/user/k/kiliakis/tomo_data/datasets_encoder_02-12-22'
data_dir = './tomo_data/datasets_encoder_TF_03-03-23'

IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8
input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

var_name = 'enEr'

# Train specific
train_cfg = {
    'epochs': 15,
    'dense_layers': [16],
    'filters': [8],
    'cropping': [0, 0],
    'kernel_size': 7,
    'strides': [2, 2],
    'activation': 'relu',
    'pooling': None,
    'pooling_size': [0, 0],
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
        'epochs': 20,
        'cropping': [0, 0],
        'filters': [2, 4, 8],
        'kernel_size': [5, 5, 5],
        'strides': [2, 2],
        'dense_layers': [1024, 512, 128],
        'activation': 'relu',
        'pooling': None,
        'dropout': 0.0,
        'lr': 1e-3,
        'normalization': 'minmax',
        'batch_size': 32
    },
}

param_space = {
    'cropping': [[0, 0]],
    'kernel_size': [[13, 9, 7], [9, 9, 9], [9, 7, 5], [7, 7, 7], [5, 5, 5]],
    'filters': [[4, 8, 16], [4, 16, 32], [8, 16, 16],
                [8, 16, 32]],
    'dense_layers': [[1024, 512, 128], [1024, 256, 128],
                     [1024, 512, 64], [1024, 256, 64], ]
}


def train_test_model(var_name, x_train, y_train, x_valid, y_valid, hparamdir, hparams):
    cfg = train_cfg.copy()
    cfg.update(model_cfg.get(var_name, {}))
    cfg.update(hparams)

    model = EncoderSingle(input_shape=input_shape,
                          output_name=var_name,
                          **cfg)
#     weights_dir = os.path.join(hparamdir, 'weights')
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
    config_dict = {}
    config_dict['encoder'] = cfg.copy()
    config_dict['encoder'].update({
        'min_train_loss': float(np.min(history.history['loss'])),
        'min_valid_loss': float(np.min(history.history['val_loss'])),
        'total_train_time': total_t,
        'used_gpus': len(gpus)
    })

    # save config_dict
#     with open(os.path.join(hparamdir, 'encoder-summary.yml'), 'w') as configfile:
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

    x_train, y_train = encoder_files_to_tensors(
        file_names, normalization=train_cfg['normalization'])

    # Repeat for validation data
    file_names = sample_files(
        VALIDATION_PATH, train_cfg['dataset%'], keep_every=num_Turns_Case)
    print('Number of Validation files: ', len(file_names))

    x_valid, y_valid = encoder_files_to_tensors(
        file_names, normalization=train_cfg['normalization'])

    end_t = time.time()
    print(
        f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')

    train = tf.gather(y_train, var_names.index(var_name), axis=1)
    valid = tf.gather(y_valid, var_names.index(var_name), axis=1)

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
        history, loss = train_test_model(
            var_name, x_train, train, x_valid, valid, os.path.join(trial_dir, run_name), hparams)
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
    fname = f'hparam_dicts/{var_name}_{timestamp}.pkl'
    with open(fname, 'wb') as handle:
        pickle.dump(overall_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
