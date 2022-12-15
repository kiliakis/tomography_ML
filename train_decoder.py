# Train the ML model

from models import Decoder
from utils import sample_files
from utils import plot_loss, decoder_files_to_tensors, load_decoder_data
import time
import glob
import shutil
import tensorflow as tf
from tensorflow import keras
import yaml
import os
import numpy as np
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

parser = argparse.ArgumentParser(description='Train the decoder models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

# Initialize parameters
data_dir = '/eos/user/k/kiliakis/tomo_data/datasets_decoder_02-12-22'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

# Data specific
DATA_LOAD_METHOD = 'TENSOR'
IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8
# BUFFER_SIZE = 32768
latent_dim = 7  # 6 + the new VrfSPS
additional_latent_dim = 1

# Train specific
train_cfg = {
    'epochs': 100,
    'dense_layers': [latent_dim + additional_latent_dim, 64, 1024],
    'filters': [32, 16, 8, 1],
    'kernel_size': 7,
    'strides': [2, 2],
    'final_kernel_size': 5,
    'activation': 'relu',
    'final_activation': 'tanh',
    'dropout': 0.,
    'loss': 'mse',
    'normalization': 'minmax',
    'lr': 1e-3,
    'dataset%': 0.1
}

if __name__ == '__main__':

    args = parser.parse_args()
    # If input config file is provided, read input from config file
    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['decoder']
        timestamp = input_config['timestamp']

    print('\n---- Configuration: ----\n')
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
        # Create the datasets
        if DATA_LOAD_METHOD=='TENSOR':
            # First the training data
            file_names = sample_files(
                TRAINING_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Training files: ', len(file_names))
            x_train, y_train = decoder_files_to_tensors(
                file_names, normalization=train_cfg['normalization'])

            # Repeat for validation data
            file_names = sample_files(
                VALIDATION_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Validation files: ', len(file_names))

            x_valid, y_valid = decoder_files_to_tensors(
                file_names, normalization=train_cfg['normalization'])
        elif DATA_LOAD_METHOD=='DATASET':
            # First the training data
            file_names = sample_files(
                TRAINING_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Training files: ', len(file_names))

            # convert to dataset
            train_dataset = tf.data.Dataset.from_tensor_slices(file_names)
            # Then map function to dataset
            # this returns pairs of tensors with shape (128, 128, 1) and (8,)
            train_dataset = train_dataset.map(lambda x: tf.py_function(
                load_decoder_data,
                [x, train_cfg['normalization']],
                [tf.float32, tf.float32]))
            # 4. Ignore errors in case they appear
            train_dataset = train_dataset.apply(
                tf.data.experimental.ignore_errors())
            # cache the dataset
            # train_dataset = train_dataset.cache(
            #     os.path.join(cache_dir, 'train_cache'))
            # batch the dataset
            train_dataset = train_dataset.batch(BATCH_SIZE)

            file_names = sample_files(
                VALIDATION_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Validation files: ', len(file_names))
            # convert to dataset
            valid_dataset = tf.data.Dataset.from_tensor_slices(file_names)
            # Then map function to dataset
            # this returns pairs of tensors with shape (128, 128, 1) and (8,)
            valid_dataset = valid_dataset.map(lambda x: tf.py_function(
                load_decoder_data,
                [x, train_cfg['normalization']],
                [tf.float32, tf.float32]))

            valid_dataset = valid_dataset.apply(
                tf.data.experimental.ignore_errors())
            # cache the dataset
            # valid_dataset = valid_dataset.cache(
            #     os.path.join(cache_dir, 'valid_cache'))
            # batch the dataset
            valid_dataset = valid_dataset.batch(BATCH_SIZE)

        end_t = time.time()
        print(
            f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')

        start_t = time.time()
        # Model instantiation
        input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

        decoder = Decoder(input_shape, **train_cfg)

        print(decoder.model.summary())
        end_t = time.time()
        print(
            f'\n---- Model has been initialized, elapsed: {end_t - start_t} ----\n')

        print('\n---- Training the decoder ----\n')

        # callbacks, save the best model, and early stop if no improvement in val_loss
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=5, restore_best_weights=True)
        save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, 'decoder.h5'),
                                                    monitor='val_loss', save_best_only=True)

        start_time = time.time()
        
        if DATA_LOAD_METHOD=='TENSOR':
            history = decoder.model.fit(
                x=x_train, y=y_train,
                epochs=train_cfg['epochs'],
                validation_data=(x_valid, y_valid),
                callbacks=[save_best],
                batch_size=BATCH_SIZE,
                verbose=0)
        elif DATA_LOAD_METHOD=='DATASET':

            history = decoder.model.fit(
                train_dataset, epochs=train_cfg['epochs'],
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
                  title='decoder Train/Validation Loss',
                  figname=os.path.join(plots_dir, 'decoder_train_valid_loss.png'))

        # save file with experiment configuration
        print('\n---- Saving a summary ----\n')

        config_dict = {}
        config_dict['decoder'] = train_cfg.copy()

        config_dict['decoder'].update({
            'epochs': len(history.history["loss"]),
            'min_train_loss': float(np.min(train_loss_l)),
            'min_valid_loss': float(np.min(valid_loss_l)),
            'total_train_time': total_time,
            'used_gpus': len(gpus)
        })

        # save config_dict
        with open(os.path.join(trial_dir, 'decoder-summary.yml'), 'w') as configfile:
            yaml.dump(config_dict, configfile, default_flow_style=False)
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
