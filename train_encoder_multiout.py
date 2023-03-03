# Train the ML model


import time
import tensorflow as tf
from tensorflow import keras
import yaml
import os
import shutil
import numpy as np
from datetime import datetime
import argparse
# import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

from models import EncoderSingle
from utils import sample_files, plot_loss, load_encoder_data
from utils import encoder_files_to_tensors

parser = argparse.ArgumentParser(description='Train the encoder/ decoder models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

# Initialize parameters
# data_dir = './tomo_data/datasets_encoder_02-12-22'
# data_dir = './tomo_data/datasets_encoder_TF_16-12-22'
data_dir = './tomo_data/datasets_encoder_TF_03-03-23'

timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

# Data specific
IMG_OUTPUT_SIZE = 128
DATA_LOAD_METHOD='TENSOR' # it can be TENSOR or DATASET
num_Turns_Case = 1
var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']

# Train specific
train_cfg = {
    'epochs': 50,
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
    'dataset%': 1,
    'normalization': 'minmax',
    'img_normalize': 'off',
    # 'loss_weights': [0, 1, 2, 3, 4, 5, 6],
    'loss_weights': [1],
    'batch_size': 32
}

model_cfg = {
    # Best phEr config --> 3.07e-6 val_loss
    'phEr': {
        'epochs': 60,
        'dense_layers': [1024, 256, 32],
        'cropping': [0, 0],
        'kernel_size': [3, 3],
        'strides': [2, 2],
        'activation': 'relu',
        'filters': [4, 8],
        'pooling': None,
        'dropout': 0.,
        'lr': 1e-3,
        'normalization': 'minmax',
        'img_normalize': 'off',
        'batch_size': 32
    },
    # Best enErr config --> 1.54e-05 val_loss
    'enEr': {
        'epochs': 60,
        'dense_layers': [1024, 256, 64],
        'cropping': [6, 6],
        'kernel_size': [3, 3, 3],
        'strides': [2, 2],
        'activation': 'relu',
        'filters': [8, 16, 32],
        'pooling': None,
        'dropout': 0.,
        'lr': 1e-3,
        'normalization': 'minmax',
        'img_normalize': 'off',
        'batch_size': 32
    },
    # Best bl config --> 1.83e-04 val_loss
    'bl': {
        'epochs': 60,
        'cropping': [12, 12],
        'filters': [8, 16, 32],
        'kernel_size': [(13, 3), (7, 3), (3, 3)],
        'strides': [2, 2],
        'dense_layers': [1024, 256, 64],
        'activation': 'relu',
        'pooling': None,
        'dropout': 0.,
        'lr': 1e-3,
        'normalization': 'minmax',
        'img_normalize': 'off',
        'batch_size': 32
    },
    # Best inten config --> 7.57e-02 val_loss
    'inten': {
        'epochs': 60,
        'cropping': [6, 6],
        'filters': [8, 16, 32],
        'kernel_size': [13, 7, 3],
        'strides': [2, 2],
        'dense_layers': [1024, 256, 64],
        'activation': 'relu',
        'pooling': None,
        'dropout': 0.,
        'lr': 1e-3,
        'normalization': 'minmax',
        'img_normalize': 'off',
        'batch_size': 32
    },
    # best Vrf config --> 2.54e-05 val loss
    'Vrf': {
        'epochs': 60,
        'cropping': [6, 6],
        'filters': [8, 16, 32],
        'kernel_size': [13, 7, 3],
        'strides': [2, 2],
        'dense_layers': [1024, 256, 64],
        'activation': 'relu',
        'pooling': None,
        'dropout': 0.,
        'lr': 1e-3,
        'normalization': 'minmax',
        'img_normalize': 'off',
        'batch_size': 32
    },
    # best mu config --> 8.00e-04 val loss
    'mu': {
        'epochs': 60,
        'cropping': [0, 0],
        'filters': [8, 16, 32],
        'kernel_size': [5, 5, 5],
        'strides': [2, 2],
        'dense_layers': [1024, 256, 64],
        'activation': 'relu',
        'pooling': None,
        'dropout': 0.0,
        'lr': 1e-3,
        'normalization': 'minmax',
        'img_normalize': 'off',
        'batch_size': 32
    },
    # best VrfSPS config --> 7.83e-04 val loss
    'VrfSPS': {
        'epochs': 60,
        'cropping': [6, 6],
        'filters': [8, 16, 32],
        'kernel_size': [5, 5, 5],
        'strides': [2, 2],
        'dense_layers': [1024, 256, 32],
        'activation': 'relu',
        'pooling': None,
        'dropout': 0.0,
        'lr': 1e-3,
        'normalization': 'minmax',
        'img_normalize': 'off',
        'batch_size': 32
    }
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

        if DATA_LOAD_METHOD=='TENSOR':
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
        elif DATA_LOAD_METHOD=='DATASET':
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
                [x, train_cfg['normalization'], True, train_cfg['img_normalize']],
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
                [x, train_cfg['normalization'], True, train_cfg['img_normalize']],
                [tf.float32, tf.float32]))
            # Ignore errors
            valid_dataset = valid_dataset.apply(
                tf.data.experimental.ignore_errors())

            # cache the dataset
            # valid_dataset = valid_dataset.cache(
            #     os.path.join(cache_dir, 'valid_cache'))

        print(
            f'\n---- Input files have been read, elapsed: {time.time() - start_t} ----\n')

        # Model instantiation
        start_t = time.time()
        input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)
        models = {}

        for i in train_cfg['loss_weights']:
            var_name = var_names[i]
            print(f'\n---- Initializing model: {var_name} ----\n')
            
            cfg = train_cfg.copy()
            cfg.update(model_cfg.get(var_name, {}))
            model_cfg[var_name] = cfg
            
            model = EncoderSingle(input_shape=input_shape, output_name=var_name, **cfg)
            print(model.model.summary())

            if DATA_LOAD_METHOD=='TENSOR':
                models[var_name] = {'model': model.model,
                                    'train': tf.gather(y_train, i, axis=1),
                                    'valid': tf.gather(y_valid, i, axis=1)}

            elif DATA_LOAD_METHOD=='DATASET':
                models[var_name] = {'model': model.model,
                                    'train': train_dataset.map(lambda x, y: (x, y[i])).batch(train_cfg['batch_size']),
                                    'valid': valid_dataset.map(lambda x, y: (x, y[i])).batch(train_cfg['batch_size'])}
        print(
            f'\n---- Models have been initialized, elapsed: {time.time() - start_t} ----\n')

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
            if DATA_LOAD_METHOD=='TENSOR':
                history = model.fit(
                    x=x_train, y=models[var_name]['train'], 
                    epochs=train_cfg['epochs'],
                    validation_data=(x_valid, models[var_name]['valid']),
                    callbacks=[save_best], 
                    batch_size=model_cfg[var_name]['batch_size'],
                    verbose=0)
            elif DATA_LOAD_METHOD=='DATASET':
                history = model.fit(
                    models[var_name]['train'], 
                    epochs=train_cfg['epochs'],
                    validation_data=models[var_name]['valid'],
                    callbacks=[save_best],
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
        max_length = np.max([len(v) for v in historyMulti.values()])
        for k, v in historyMulti.items():
            if len(v) < max_length:
                v = v + [v[-1]] * (max_length - len(v))
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
