# Train the ML model

from models import Encoder, EncoderFunc, EncoderMulti
from utils import sample_files
from utils import plot_loss, load_encoder_data
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
data_dir = '/eos/user/k/kiliakis/tomo_data/datasets'
#data_dir = '/eos/kiliakis/tomo_data/datasets'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

# Data specific
IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8

num_Turns_Case = 50+1
var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']

# Train specific
train_cfg = {
    'epochs': 5,
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
    'dataset%': 0.1,
    'normalization': 'minmax',
    'loss_weights': [0, 1, 2, 3, 4, 5, 6]

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
        # shuffle the dataset
        # batch the dataset

        # 6. Divide dataset in batces
        # train_dataset = train_dataset.batch(BATCH_SIZE)

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
        # batch the dataset
        # valid_dataset = valid_dataset.batch(BATCH_SIZE)

        end_t = time.time()
        print(
            f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')

        start_t = time.time()
        # Model instantiation
        input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

        encoderMulti = EncoderMulti(input_shape=input_shape, **train_cfg)

        # for model in encoderMulti.models:
        #     print(model.summary())
        end_t = time.time()

        print(
            f'\n---- Model has been initialized, elapsed: {end_t - start_t} ----\n')

        models = {}

        for i in train_cfg['loss_weights']:
            var_name = var_names[i]
            models[var_name] = {'model': encoderMulti.models[var_name],
                                'train': train_dataset.map(lambda x, y: (x, y[i])).batch(BATCH_SIZE),
                                'valid': valid_dataset.map(lambda x, y: (x, y[i])).batch(BATCH_SIZE)}
            print(encoderMulti.models[var_name].summary())

        historyMulti = {}
        for var_name in models:
            model = models[var_name]['model']

            # Train the encoder

            print(f'\n---- {var_name}: Training the encoder ----\n')

            start_time = time.time()

            # callbacks, save the best model, and early stop if no improvement in val_loss
            stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=5, restore_best_weights=True)
            save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, f'encoder_{var_name}.h5'),
                                                        monitor='val_loss', save_best_only=True)

            history = model.fit(
                models[var_name]['train'], epochs=train_cfg['epochs'],
                validation_data=models[var_name]['valid'],
                callbacks=[stop_early, save_best],
                verbose=0)
            historyMulti[f'{var_name}_loss'] = history.history['loss']
            historyMulti[f'{var_name}_val_loss'] = history.history['val_loss']
            
            total_time = time.time() - start_time
            print(
                f'\n---- {var_name}: Training complete, epochs: {len(history.history["loss"])}, min loss {np.min(history.history["val_loss"])}, total time {total_time} ----\n')


        # Plot training and validation loss
        print('\n---- Plotting loss ----\n')

        train_loss_l = []
        valid_loss_l = []
        for k, v in historyMulti.items():
            if 'val' in k:
                valid_loss_l.append(v)
            else:
                train_loss_l.append(v)
        
        train_loss_l = np.mean(train_loss_l, axis=0)
        valid_loss_l = np.mean(valid_loss_l, axis=0)
        print(train_loss_l)
        print(valid_loss_l)

        plot_loss({'training': train_loss_l, 'validation': valid_loss_l},
                  title='Encoder Train/Validation Loss',
                  figname=os.path.join(plots_dir, 'encoder_train_valid_loss.png'))

        plot_loss(historyMulti, title='Encoder loss per output',
                  figname=os.path.join(plots_dir, 'encoder_per_output_loss.png'))

        print('\n---- Saving a summary ----\n')

        # save file with experiment configuration
        config_dict = {}
        config_dict['encoder'] = train_cfg.copy()
        config_dict['encoder'].update({
            'epochs': len(history.history["loss"]),
            'total_train_time': total_time,
            'used_gpus': len(gpus),
            'min_train_loss': float(np.min(train_loss_l)),
            'min_valid_loss': float(np.min(valid_loss_l)),
        })

        config_dict['encoder']['losses'] = {}
        for k, v in historyMulti.items():
            config_dict['encoder']['losses'][k] = float(np.min(v))

        # save config_dict
        with open(os.path.join(trial_dir, 'encoder-summary.yml'), 'w') as configfile:
            yaml.dump(config_dict, configfile, default_flow_style=False)

    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
