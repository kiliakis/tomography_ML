# Train the ML model

from mlp_lhc_tomography.utils import sample_files
from local_models import Tomoscope
from local_utils import plot_loss, tomoscope_files_to_tensors, load_tomoscope_data
from local_utils import fast_tensor_load_encdec
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

parser = argparse.ArgumentParser(description='Train the tomoscope models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

# Initialize parameters
data_dir = './tomo_data/datasets_tomoscope_TF_24-03-23'


timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

# Data specific
DATA_LOAD_METHOD = 'FAST_TENSOR' # Can be TENSOR, FAST_TENSOR or DATASET
IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8

# Train specific
train_cfg = {
    'epochs': 10, 'output_turns': 10,
    'cropping': [0, 0],
    'enc_dense_layers': [1024, 256],
    'dec_dense_layers': [1024],
    'enc_filters': [4, 8, 16],
    'dec_filters': [8, 16, 10],
    'enc_kernel_size': 3,
    'dec_kernel_size': 3,
    'enc_strides': [2, 2],
    'dec_strides': [2, 2],
    'enc_activation': 'relu',
    'dec_activation': 'relu',
    'enc_pooling': None, 'dec_pooling': None,
    'enc_pooling_size': [0, 0], 'dec_pooling_size': [0, 0],
    'enc_pooling_strides': [1, 1], 'dec_pooling_strides': [1, 1],
    'enc_pooling_padding': 'valid', 'dec_pooling_padding': 'valid',
    'enc_dropout': 0.0, 'dec_dropout': 0.0,
    'metrics': [], 'use_bias': False, 'batchnorm': False,
    'learning_rate': 1e-3,
    'dataset%': 0.1,
    'normalization': 'minmax', 'img_normalize': 'off',
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
        train_cfg = input_config['tomoscope']
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
    # os.makedirs(cache_dir, exist_ok=True)

    try:
        start_t = time.time()
        # Create the datasets
        if DATA_LOAD_METHOD=='TENSOR':
            # First the training data
            file_names = sample_files(
                TRAINING_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Training files: ', len(file_names))
            x_train, y_train = tomoscope_files_to_tensors(
                file_names, normalization=train_cfg['normalization'],
                img_normalize=train_cfg['img_normalize'],
                ps_normalize=train_cfg['ps_normalize'], num_turns=train_cfg['output_turns'])

            # Repeat for validation data
            file_names = sample_files(
                VALIDATION_PATH, train_cfg['dataset%'], keep_every=1)
            print('Number of Validation files: ', len(file_names))

            x_valid, y_valid = tomoscope_files_to_tensors(
                file_names, normalization=train_cfg['normalization'],
                img_normalize=train_cfg['img_normalize'],
                ps_normalize=train_cfg['ps_normalize'], num_turns=train_cfg['output_turns'])
            
            print('x_train shape: ', x_train.shape)
            print('x_valid shape: ', x_valid.shape)
        
        elif DATA_LOAD_METHOD == 'FAST_TENSOR':
            assert train_cfg['normalization'] == 'minmax'
            # assert train_cfg['ps_normalize'] == 'off'
            assert train_cfg['img_normalize'] == 'off'

            TRAINING_PATH = os.path.join(ML_dir, 'tomoscope-training-??.npz')
            VALIDATION_PATH = os.path.join(ML_dir, 'tomoscpe-validation-??.npz')

            x_train, turn_train, latent_train, y_train = fast_tensor_load_encdec(
                TRAINING_PATH, train_cfg['dataset%'])
            print('Number of Training files: ', len(y_train))

            x_valid, turn_valid, latent_valid, y_valid = fast_tensor_load_encdec(
                VALIDATION_PATH, train_cfg['dataset%'])
            print('Number of Validation files: ', len(y_valid))

        elif DATA_LOAD_METHOD=='DATASET':
            exit('DATASET method not supported')

        end_t = time.time()
        print(
            f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')

        start_t = time.time()
        # Model instantiation
        input_shape = x_train.shape[1:]

        import keras.backend as K

        def custom_loss(ps_true, ps_pred):
            """Custom loss function that recreates the WF from the PS and compares them.

            Args:
                ps_true (_type_): The true PS with dim (128, 128, output_turns)
                ps_pred (_type_): The predicted PS with dim (128, 128, output_turns)

            Returns:
                _type_: _description_
            """
            # print(ps_pred.shape, ps_true.shape)

            wf_pred = K.sum(ps_pred, axis=1)
            wf_true = K.sum(ps_true, axis=1)
            # print(wf_pred.shape, wf_true.shape)
            loss = K.mean(K.square(wf_true - wf_pred))
            return loss

        if train_cfg['loss'] == 'custom':
            train_cfg['loss'] = custom_loss

        tomoscope = Tomoscope(input_shape=input_shape, **train_cfg)

        print(tomoscope.model.summary())
        end_t = time.time()
        print(
            f'\n---- Model has been initialized, elapsed: {end_t - start_t} ----\n')

        print('\n---- Training the model ----\n')

        # callbacks, save the best model, and early stop if no improvement in val_loss
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=10, restore_best_weights=True)
        save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, 'tomoscope.h5'),
                                                    monitor='val_loss', save_best_only=True)

        start_time = time.time()
        
        if 'TENSOR' in DATA_LOAD_METHOD:
            history = tomoscope.model.fit(
                x=x_train, y=y_train,
                epochs=train_cfg['epochs'],
                validation_data=(x_valid, y_valid),
                callbacks=[save_best],
                batch_size=BATCH_SIZE,
                verbose=0)
        elif DATA_LOAD_METHOD=='DATASET':
            exit('DATASET dataload method not supported')
            # history = tomoscope.model.fit(
            #     train_dataset, epochs=train_cfg['epochs'],
            #     validation_data=valid_dataset,
            #     callbacks=[save_best],
            #     verbose=0)

        total_time = time.time() - start_time
        print(
            f'\n---- Training complete, epochs: {len(history.history["loss"])}, min loss {np.min(history.history["val_loss"])}, total time {total_time} ----\n')

        # Plot training and validation loss
        print('\n---- Plotting loss ----\n')

        train_loss_l = np.array(history.history['loss'])
        valid_loss_l = np.array(history.history['val_loss'])

        plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
                title='Tomoscpe Train/Validation Loss',
                figname=os.path.join(plots_dir, 'tomoscope_train_valid_loss.png'))

        # save file with experiment configuration
        print('\n---- Saving a summary ----\n')

        config_dict = {}
        config_dict['tomoscope'] = train_cfg.copy()

        config_dict['tomoscope'].update({
            'epochs': len(history.history["loss"]),
            'min_train_loss': float(np.min(train_loss_l)),
            'min_valid_loss': float(np.min(valid_loss_l)),
            'total_train_time': total_time,
            'used_gpus': len(gpus)
        })

        # save config_dict
        with open(os.path.join(trial_dir, 'tomoscope-summary.yml'), 'w') as configfile:
            yaml.dump(config_dict, configfile, default_flow_style=False)
    finally:
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
