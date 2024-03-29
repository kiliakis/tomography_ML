#!/usr/bin/env python
# coding: utf-8


# Train the ML model
from utils import fast_tensor_load, plot_loss, plot_sample, visualize_weights, get_model_size

import time
import tensorflow as tf
from tensorflow import keras
import yaml
import os
import numpy as np
from datetime import datetime
import argparse

# from tensorflow.keras import mixed_precision
# mixed_precision.set_global_policy('mixed_float16')


import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

parser = argparse.ArgumentParser(description='Train the encoder/ decoder models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')


IMG_OUTPUT_SIZE = 128
DATA_LOAD_METHOD='FAST_TENSOR' # it can be TENSOR or DATASET or FAST_TENSOR

num_Turns_Case = 1
var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']
# Initialize parameters
# data_dir = './tomo_data/datasets_encoder_TF_24-03-23'
data_dir = './tomo_data/datasets_encoder_TF_08-11-23'
data_type = 'float32'

# data_dir = './tomo_data/datasets'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
# print('Using timestamp: ', timestamp)

# Train specific
train_cfg = {
    'epochs': 50,
    'filters': [8, 16, 32],
    'dense_layers': [512, 256],
    'decoder_dense_layers': [512],
    'cropping': [14, 14],
    'kernel_size': 3,
    'strides': [2, 2],
    'activation': 'relu',
    'conv_padding': 'same',
    'pooling': None,
    'pooling_size': [0, 0],
    'pooling_strides': [1, 1],
    'pooling_padding': 'valid',
    'dropout': 0.0,
    'loss': 'mae',
    'lr': 1e-3,
    'dataset%': 1,
    'use_bias': False,
    'normalization': 'minmax',
    'img_normalize': 'off',
    'conv_batchnorm': False,
    'dense_batchnorm': False,
    'batch_size': 128
}

autoenc_class = 'AutoEncoderTranspose'


dense_layers = [[1024, 16], [512, 16]]

# dense_layers = [[1024, 512, 256], [1024, 256, 128], [1024, 256, 64]]


def calculate_padding(input_shape, target_shape):
    # Calculate the padding needed for the first two dimensions
    padding_first_dim = (target_shape[0] - input_shape[0]) // 2
    mod_first_dim = (target_shape[0] - input_shape[0]) % 2
    padding_second_dim = (target_shape[1] - input_shape[1]) // 2
    mod_second_dim = (target_shape[1] - input_shape[1]) % 2

    # If the padding doesn't divide evenly, add the extra padding to one side
    pad_first_dim_left = padding_first_dim + mod_first_dim
    pad_second_dim_left = padding_second_dim + mod_second_dim

    # Create the padding configuration for np.pad
    padding_config = (
        # Padding for the first dimension
        (pad_first_dim_left, padding_first_dim),
        # Padding for the second dimension
        (pad_second_dim_left, padding_second_dim),
        # (0, 0)  # Padding for the third dimension
    )

    return padding_config


# model definition
class AutoEncoderTranspose(keras.Model):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name='autoencoder',
                 input_shape=(128, 128, 1), dense_layers=[7],
                 decoder_dense_layers=[7],
                 cropping=[[0, 0], [0, 0]],
                 filters=[8, 16, 32],  kernel_size=3, conv_padding='same',
                 strides=[2, 2], activation='relu',
                 final_activation='linear', final_kernel_size=3,
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='valid',
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=True, conv_batchnorm=False,
                 dense_batchnorm=False, data_type='float32',
                 **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

        # set the input size
        inputs = keras.Input(shape=input_shape, name='Input', dtype=data_type)
        # this is the autoencoder case
        # crop the edges
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='Crop')(inputs)
        x = cropped

        # For evey Convolutional layer
        for i, f in enumerate(filters):
            # Add the Convolution
            x = keras.layers.Conv2D(
                filters=f, kernel_size=kernel_size[i], strides=strides[i],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_{i+1}')(x)

            # Apply batchnormalization
            if conv_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Apply the activation function
            x = keras.activations.get(activation)(x)

            # Optional pooling after the convolution
            if pooling == 'Max':
                x = keras.layers.MaxPooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'MaxPooling_{i+1}')(x)
            elif pooling == 'Average':
                x = keras.layers.AveragePooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'AveragePooling_{i+1}')(x)

        # we have reached the latent space
        last_shape = x.shape[1:]
        x = keras.layers.Flatten(name='Flatten')(x)
        flat_shape = x.shape[1:]
        # Now we add the dense layers
        for i, units in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'encoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # a dummy layer just to name it latent space
        x = keras.layers.Lambda(lambda x: x, name='LatentSpace')(x)

        # Now we add the decoder dense layers
        for i, units in enumerate(decoder_dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'decoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # Now reshape back to last_shape
        x = keras.layers.Dense(units=np.prod(flat_shape), activation=activation,
                               name='decoder_dense_final')(x)

        x = keras.layers.Reshape(target_shape=last_shape, name='Reshape')(x)
        # Now with transpose convolutions we go back to the original size

        for i, f in enumerate(filters[::-1]):
            x = keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size[-i -
                                                   1], strides=strides[-i-1],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_Transpose_{i+1}')(x)

        # final convolution to get the right number of channels
        x = keras.layers.Conv2DTranspose(filters=1, kernel_size=final_kernel_size,
                                         strides=1, use_bias=use_bias, padding='same',
                                         name=f'CNN_Transpose_Final')(x)

        x = keras.layers.Activation(activation=final_activation,
                                    name='final_activation')(x)
        before_padding = x
        # Add zero padding
        padding = calculate_padding(
            input_shape=before_padding.shape[1:], target_shape=input_shape)
        outputs = keras.layers.ZeroPadding2D(
            padding=padding, name='Padding')(x)

        model = keras.Model(inputs=inputs, outputs=outputs, name=output_name)

        # assert model.layers[-1].output_shape[1:] == input_shape

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model

    def predict(self, wf_in):
        wf_out = self.model(wf_in)
        return wf_out

    def load(self, weights_file):
        self.model = keras.models.load_model(weights_file)

    def save(self, weights_file):
        self.model.save(weights_file)


# model definition
class AutoEncoderTransposeIB(keras.Model):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name='autoencoder',
                 input_shape=(128, 128, 1), dense_layers=[7],
                 decoder_dense_layers=[],
                 cropping=[[0, 0], [0, 0]],
                 filters=[8, 16, 32],  kernel_size=3, conv_padding='same',
                 strides=[2, 2], activation='relu',
                 final_activation='linear', final_kernel_size=3,
                 pooling=None, pooling_size=[2, 2],
                 pooling_strides=[1, 1], pooling_padding='valid',
                 dropout=0.0, learning_rate=0.001, loss='mse',
                 metrics=[], use_bias=True, conv_batchnorm=False,
                 dense_batchnorm=False, data_type='float32',
                 **kwargs):
        super().__init__()

        self.output_name = output_name
        self.inputShape = input_shape
        # img_input =
        # the kernel_size can be a single int or a list of ints
        if isinstance(kernel_size, int):
            kernel_size = [kernel_size] * len(filters)
        assert len(kernel_size) == len(filters)

        # the strides can be a list of two ints, or a list of two-int lists
        if isinstance(strides[0], int):
            strides = [strides for _ in filters]
        assert len(strides) == len(filters)

        # set the input size
        img_input = keras.Input(
            shape=input_shape, name='Input', dtype=data_type)
        scalar_input = keras.Input(shape=(1,), name='Scalar_input')
        # this is the autoencoder case
        # crop the edges
        cropped = keras.layers.Cropping2D(
            cropping=cropping, name='Crop')(img_input)
        x = cropped

        # For evey Convolutional layer
        for i, f in enumerate(filters):
            # Add the Convolution
            x = keras.layers.Conv2D(
                filters=f, kernel_size=kernel_size[i], strides=strides[i],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_{i+1}')(x)

            # Apply batchnormalization
            if conv_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Apply the activation function
            x = keras.activations.get(activation)(x)

            # Optional pooling after the convolution
            if pooling == 'Max':
                x = keras.layers.MaxPooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'MaxPooling_{i+1}')(x)
            elif pooling == 'Average':
                x = keras.layers.AveragePooling2D(
                    pool_size=pooling_size, strides=pooling_strides,
                    padding=pooling_padding, name=f'AveragePooling_{i+1}')(x)

        # we have reached the latent space
        last_shape = x.shape[1:]
        x = keras.layers.Flatten(name='Flatten')(x)
        flat_shape = x.shape[1:]
        # Now we add the dense layers
        for i, units in enumerate(dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'encoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # append the intensity to x
        x = keras.layers.Concatenate(
            axis=-1, name='Concatenate')([x, scalar_input])

        # a dummy layer just to name it latent space
        x = keras.layers.Lambda(lambda x: x, name='LatentSpace')(x)

        # Now we add the decoder dense layers
        for i, units in enumerate(decoder_dense_layers):
            # Add the layer
            x = keras.layers.Dense(units=units, activation=activation,
                                   name=f'decoder_dense_{i+1}')(x)

            # Apply batchnormalization
            if dense_batchnorm:
                x = tf.keras.layers.BatchNormalization()(x)

            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)

        # Now reshape back to last_shape
        x = keras.layers.Dense(units=np.prod(flat_shape), activation=activation,
                               name='decoder_dense_final')(x)

        x = keras.layers.Reshape(target_shape=last_shape, name='Reshape')(x)
        # Now with transpose convolutions we go back to the original size

        for i, f in enumerate(filters[::-1]):
            x = keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size[-i -
                                                   1], strides=strides[-i-1],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_Transpose_{i+1}')(x)

        # final convolution to get the right number of channels
        x = keras.layers.Conv2DTranspose(filters=1, kernel_size=final_kernel_size,
                                         strides=1, use_bias=use_bias, padding='same',
                                         name=f'CNN_Transpose_Final')(x)

        x = keras.layers.Activation(activation=final_activation,
                                    name='final_activation')(x)
        before_padding = x
        # Add zero padding
        padding = calculate_padding(
            input_shape=before_padding.shape[1:], target_shape=input_shape)
        outputs = keras.layers.ZeroPadding2D(
            padding=padding, name='Padding')(x)

        model = keras.Model(
            inputs=[img_input, scalar_input], outputs=outputs, name=output_name)

        # assert model.layers[-1].output_shape[1:] == input_shape

        # Also initialize the optimizer and compile the model
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model = model

    def predict(self, wf_in):
        wf_out = self.model(wf_in)
        return wf_out

    def load(self, weights_file):
        self.model = keras.models.load_model(weights_file)

    def save(self, weights_file):
        self.model.save(weights_file)


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

    # Initialize directories
    trial_dir = os.path.join('./trials/', timestamp)
    weights_dir = os.path.join(trial_dir, 'weights')
    plots_dir = os.path.join(trial_dir, 'plots')

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
    assert os.path.exists(ML_dir)

    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')

    # create the directory to store the results
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    np.random.seed(42)

    start_t = time.time()

    assert train_cfg['normalization'] == 'minmax'
    assert train_cfg['img_normalize'] == 'off'

    TRAINING_PATH = os.path.join(ML_dir, 'training-??.npz')
    VALIDATION_PATH = os.path.join(ML_dir, 'validation-??.npz')

    x_train, y_train = fast_tensor_load(TRAINING_PATH, train_cfg['dataset%'], dtype=data_type)
    print('Number of Training files: ', len(y_train))

    x_valid, y_valid = fast_tensor_load(
        VALIDATION_PATH, train_cfg['dataset%'], dtype=data_type)
    print('Number of Validation files: ', len(y_valid))

    end_t = time.time()
    print(
        f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')

    config_dict = {}
    config_dict['model'] = train_cfg.copy()
    config_dict['timestamp'] = timestamp
    # callbacks, save the best model, and early stop if no improvement in val_loss
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=10, restore_best_weights=True)

    for dense_units in dense_layers:
        decoder_dense_layers = dense_units[::-1][1:]
        dense_units_str = '_'.join([str(d) for d in dense_units])
        save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, f'autoenc_dense_{dense_units_str}.h5'),
                                                    monitor='val_loss', save_best_only=True)

        train_cfg['dense_layers'] = dense_units
        train_cfg['decoder_dense_layers'] = decoder_dense_layers
        # autoenc = AutoEncoderTranspose(**train_cfg)
        if autoenc_class == 'AutoEncoderTranspose':
            autoenc = AutoEncoderTranspose(**train_cfg)
            x_train_data = x_train
            x_valid_data = x_valid
        else:
            autoenc = AutoEncoderTransposeIB(**train_cfg)
            x_train_data = [x_train, y_train[:, 3]]
            x_valid_data = [x_valid, y_valid[:, 3]]


        print(f"----- Model Initialized, size: {get_model_size(autoenc.model):.1f}MB, training with dense units: ", dense_units)
        start_time = time.time()

        history = autoenc.model.fit(
            x=x_train_data, y=x_train, 
            epochs=train_cfg['epochs'],
            validation_data=(x_valid_data, x_valid),
            callbacks=[stop_early, save_best],
            batch_size=train_cfg['batch_size'],
            verbose=1)

        total_time = time.time() - start_time
        print(
                f'\n---- {timestamp} {dense_units_str}: Training complete, epochs: {len(history.history["loss"])}, min loss {np.min(history.history["val_loss"])}, total time {total_time} ----\n')

        # Plot training and validation loss
        print('\n---- Plotting loss ----\n')

        train_loss_l = np.array(history.history['loss'])
        valid_loss_l = np.array(history.history['val_loss'])

        plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
                    title='Encoder Train/Validation Loss',
                    figname=os.path.join(plots_dir, f'units{dense_units_str}_train_valid_loss.png'))

        # Plot some samples
        if autoenc_class == 'AutoEncoderTransposeIB':
            x_valid = x_valid_data[0]

        plot_sample(x_valid_data, samples=[0, 1, 2, 3], autoenc=autoenc,
                    figname=os.path.join(plots_dir, f'units{dense_units_str}_samples.png'))

        # Visualize the weights
        visualize_weights(timestamp, f'autoenc_dense_{dense_units_str}.h5', prefix=dense_units_str)

        # save file with experiment configuration
        config_dict[f'{dense_units_str}'] = {
            'epochs': len(history.history["loss"]),
            'min_train_loss': float(np.min(train_loss_l)),
            'min_valid_loss': float(np.min(valid_loss_l)),
            'total_train_time': total_time,
            'used_gpus': len(gpus)
        }

    # save config_dict
    with open(os.path.join(trial_dir, 'autoencoder-summary.yml'), 'w') as configfile:
        yaml.dump(config_dict, configfile, default_flow_style=False)

    # create a plot that shows the min loss per dense units
    # min_train_loss = np.array([config_dict[d]['min_train_loss'] for d in config_dict.keys()])
    # min_valid_loss = np.array([config_dict[d]['min_valid_loss'] for d in config_dict.keys()])
    # fig = plt.figure(figsize=(8, 6))
    # dense_layers_str = ['_'.join([str(d) for d in dense_units]) for dense_units in dense_layers]
    # plt.plot(dense_layers_str, min_train_loss, label='Training')
    # plt.plot(dense_layers_str, min_valid_loss, label='Validation')
    # plt.xlabel('Dense units')
    # plt.ylabel('Min loss')
    # plt.legend()
    # plt.savefig(os.path.join(plots_dir, f'min_loss_dense_units.png'), dpi=300)
