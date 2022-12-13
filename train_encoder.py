# Train the ML model

from models import Encoder
from utils import sample_files
from utils import plot_loss, encoder_files_to_tensors, load_encoder_data
import time
import glob
import tensorflow as tf
from tensorflow import keras
import yaml
import os
import shutil
import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

parser = argparse.ArgumentParser(description='Train the encoder/ decoder models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

# Initialize parameters
data_dir = '/eos/user/k/kiliakis/tomo_data/datasets_encoder_02-12-22'
#data_dir = '/eos/kiliakis/tomo_data/datasets'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

DATA_LOAD_METHOD = 'TENSOR'  # it can be TENSOR or DATASET
# Data specific
IMG_OUTPUT_SIZE = 128
# BATCH_SIZE = 32  # 8
# BUFFER_SIZE = 32768
BUFFER_SIZE = 256

latent_dim = 7  # 6 + the new VrfSPS
num_Turns_Case = 1

# Train specific
train_cfg = {
    'epochs': 50,
    'dense_layers': [1024, 256, 64, 7],
    'filters': [8, 16, 32],
    'cropping': [0, 0],
    'kernel_size': 5,
    'strides': [2, 2],
    'activation': 'relu',
    'pooling': None,
    'pooling_size': [0, 0],
    'pooling_strides': [1, 1],
    'pooling_padding': 'valid',
    'dropout': 0.0,
    'loss': 'mse',
    'lr': 1e-3,
    'dataset%': 1,
    'normalization': 'minmax',
    'loss_weights': [1, 1, 1, 1, 1, 1, 1],
    'batch_size': 32
}

if __name__ == '__main__':

    args = parser.parse_args()
    # If input config file is provided, read input from config file
    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['encoder']
        # cnn_filters = input_config['cnn_filters']
        # dataset_keep_percent = input_config['dataset_keep_percent']
        timestamp = input_config['timestamp']
    
    print('Configuration:')
    for k, v in train_cfg.items():
        print(k, v)

    # Initialize directories
    trial_dir = os.path.join('./trials/', timestamp)
    weights_dir = os.path.join(trial_dir, 'weights')
    plots_dir = os.path.join(trial_dir, 'plots')
    cache_dir = os.path.join(trial_dir, 'cache')

    print('\n---- Using directory: ', trial_dir, ' ----\n')

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

    # Initialize train/ test / validation paths
    ML_dir = os.path.join(data_dir, 'ML_data')
    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')
    assert os.path.exists(TRAINING_PATH)
    assert os.path.exists(VALIDATION_PATH)

    # create the directory to store the results
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)
    try:
        start_t = time.time()

        if DATA_LOAD_METHOD == 'TENSOR':
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
        elif DATA_LOAD_METHOD == 'DATASET':
            # Create the datasets
            # 1. Randomly select the training data
            file_names = sample_files(
                TRAINING_PATH, train_cfg['dataset%'], keep_every=num_Turns_Case)
            print('Number of Training files: ', len(file_names))

            # 2. Convert to tensor dataset
            train_dataset = tf.data.Dataset.from_tensor_slices(file_names)

            # 3. Then map function to dataset
            # this returns pairs of tensors with shape (128, 128, 1) and (7,)
            train_dataset = train_dataset.map(lambda x: tf.py_function(
                load_encoder_data,
                [x, train_cfg['normalization'], True],
                [tf.float32, tf.float32]))

            # 4. Ignore errors in case they appear
            train_dataset = train_dataset.apply(
                tf.data.experimental.ignore_errors())

            # 5. Optionally cache the dataset
            # train_dataset = train_dataset.cache(
            #     os.path.join(cache_dir, 'train_cache'))

            # Repeat for validation data
            file_names = sample_files(
                VALIDATION_PATH, train_cfg['dataset%'], keep_every=num_Turns_Case)
            print('Number of Validation files: ', len(file_names))
            # convert to dataset
            valid_dataset = tf.data.Dataset.from_tensor_slices(file_names)
            # Then map function to dataset
            # this returns pairs of tensors with shape (128, 128, 1) and (7,)
            valid_dataset = valid_dataset.map(lambda x: tf.py_function(
                load_encoder_data,
                [x, train_cfg['normalization'], True],
                [tf.float32, tf.float32]))
            # Ignore errors
            valid_dataset = valid_dataset.apply(
                tf.data.experimental.ignore_errors())

            # cache the dataset
            # valid_dataset = valid_dataset.cache(
            #     os.path.join(cache_dir, 'valid_cache'))

        print(
            f'\n---- Input files have been read, elapsed: {time.time() - start_t} ----\n')

        start_t = time.time()
        # Model instantiation
        input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

        encoder = Encoder(input_shape=input_shape, **train_cfg)

        print(encoder.model.summary())
        end_t = time.time()

        print(
            f'\n---- Model has been initialized, elapsed: {end_t - start_t} ----\n')

        # Train the encoder
        print('\n---- Training the encoder ----\n')

        # callbacks, save the best model, and early stop if no improvement in val_loss
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5, restore_best_weights=True)
        save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, 'encoder.h5'),
                                                    monitor='val_loss', save_best_only=True)

        start_time = time.time()
        if DATA_LOAD_METHOD == 'TENSOR':
            history = encoder.model.fit(
                x=x_train, y=y_train,
                epochs=train_cfg['epochs'],
                validation_data=(x_valid, y_valid),
                callbacks=[save_best],
                batch_size=train_cfg['batch_size'],
                verbose=0)
        elif DATA_LOAD_METHOD == 'DATASET':
            history = encoder.model.fit(
                train_dataset,
                epochs=train_cfg['epochs'],
                validation_data=valid_dataset,
                callbacks=[save_best],
                verbose=0)

        total_time = time.time() - start_time
        print(
            f'\n---- Training complete, epochs: {len(history.history["loss"])}, total time {total_time} ----\n')

        # Plot training and validation loss
        print('\n---- Plotting loss ----\n')

        train_loss_l = np.array(history.history['loss'])
        valid_loss_l = np.array(history.history['val_loss'])

        plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
                  title='Encoder Train/Validation Loss',
                  figname=os.path.join(plots_dir, 'encoder_train_valid_loss.png'))

        print('\n---- Saving a summary ----\n')

        # save file with experiment configuration
        config_dict = {}
        config_dict['encoder'] = train_cfg.copy()
        config_dict['encoder'].update({
            'epochs': len(history.history["loss"]),
            'min_train_loss': float(np.min(train_loss_l)),
            'min_valid_loss': float(np.min(valid_loss_l)),
            'total_train_time': total_time,
            'used_gpus': len(gpus)
        })

        # save config_dict
        with open(os.path.join(trial_dir, 'encoder-summary.yml'), 'w') as configfile:
            yaml.dump(config_dict, configfile, default_flow_style=False)

    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
