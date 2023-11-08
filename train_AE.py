#!/usr/bin/env python
# coding: utf-8


# Train the ML model
from utils import fast_tensor_load, plot_loss

import time
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
data_dir = './tomo_data/datasets_encoder_TF_24-03-23'

# data_dir = './tomo_data/datasets'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
# print('Using timestamp: ', timestamp)

# Train specific
train_cfg = {
    'epochs': 50,
    'filters': [8, 16, 32],
    'cropping': [0, 0],
    'kernel_size': 3,
    'strides': [2, 2],
    'activation': 'relu',
    'conv_padding': 'same',
    'pooling': None,
    'pooling_size': [0, 0],
    'pooling_strides': [1, 1],
    'pooling_padding': 'valid',
    'dropout': 0.1,
    'loss': 'mse',
    'lr': 1e-3,
    'dataset%': 1,
    'use_bias': True,
    'normalization': 'minmax',
    'img_normalize': 'off',
    'batch_size': 128
}

dense_layers = [1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

# dense_layers = [ 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]


def plot_sample(x_valid, samples, autoenc, figname=None):

    ncols = len(samples)
    # Get nrows * nrows random images
    # sample = np.random.choice(np.arange(len(x_train)),
    #                         size=ncols, replace=False)

    samples_X = tf.gather(x_valid, samples)
    pred_samples_X = autoenc.predict(samples_X)
    # samples_y = tf.gather(y_train, sample)

    # Create 3x3 grid of figures
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(14, 7))
    # axes = np.ravel(axes)
    for i in range(ncols):
        ax = axes[0, i]
        ax.set_xticks([])
        ax.set_yticks([])
        # show the image
        ax.imshow(samples_X[i], cmap='jet')
        # Set the label
        ax.set_title(f'Real')

        ax = axes[1, i]
        ax.set_xticks([])
        ax.set_yticks([])
        # show the image
        ax.imshow(pred_samples_X[i], cmap='jet')
        # Set the label
        ax.set_title(f'Pred, MSE: {np.mean((samples_X[i] - pred_samples_X[i])**2):.2e}')
    
    if figname is not None:
        plt.savefig(figname, dpi=300)
    else:
        plt.show()
    plt.close()

# model definition
class AutoEncoder(keras.Model):
    # Pooling can be None, or 'Average' or 'Max'
    def __init__(self, output_name='autoencoder',
                input_shape=(128, 128, 1), dense_layers=[7],
                cropping=[[0, 0], [0, 0]], 
                filters=[8, 16, 32],  kernel_size=3, conv_padding='same', 
                strides=[2, 2], activation='relu',
                final_activation='linear', final_kernel_size=3,
                pooling=None, pooling_size=[2, 2],
                pooling_strides=[1, 1], pooling_padding='valid',
                dropout=0.0, learning_rate=0.001, loss='mse',
                metrics=[], use_bias=True, batchnorm=False,
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
        inputs = keras.Input(shape=input_shape, name='Input')
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
            if batchnorm:
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
                                name=f'dense_{i+1}')(x)
            # Add dropout optionally
            if dropout > 0 and dropout < 1:
                x = keras.layers.Dropout(dropout, name=f'dropout_{i+1}')(x)
        # Now reshape back to last_shape
        x = keras.layers.Dense(units=np.prod(flat_shape), activation=activation,
                                name='dense_final')(x)
        x = keras.layers.Reshape(target_shape=last_shape, name='Reshape')(x)
        # Now with transpose convolutions we go back to the original size

        for i, f in enumerate(filters[::-1]):
            x = keras.layers.Conv2DTranspose(
                filters=f, kernel_size=kernel_size[-i-1], strides=strides[-i-1],
                use_bias=use_bias, padding=conv_padding,
                name=f'CNN_Transpose_{i+1}')(x)
        
        # final convolution to get the right number of channels
        x = keras.layers.Conv2DTranspose(filters=1, kernel_size=final_kernel_size, 
                strides=1, use_bias=use_bias, padding='same',
                name=f'CNN_Transpose_Final')(x)

        outputs = keras.layers.Activation(activation=final_activation, 
                    name='final_activation')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        assert model.layers[-1].output_shape[1:] == input_shape

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

    x_train, y_train = fast_tensor_load(TRAINING_PATH, train_cfg['dataset%'])
    print('Number of Training files: ', len(y_train))

    x_valid, y_valid = fast_tensor_load(VALIDATION_PATH, train_cfg['dataset%'])
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
        save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, f'autoencoder_dense_{dense_units}.h5'),
                                                    monitor='val_loss', save_best_only=True)

        train_cfg['dense_layers'] = [dense_units]
        autoenc = AutoEncoder(**train_cfg)

        print("----- Model Initialized, training with dense units: ", dense_units)
        start_time = time.time()

        history = autoenc.model.fit(
            x=x_train, y=x_train, 
            epochs=train_cfg['epochs'],
            validation_data=(x_valid, x_valid),
            callbacks=[stop_early], 
            batch_size=train_cfg['batch_size'],
            verbose=1)

        total_time = time.time() - start_time
        print(
                f'\n---- {dense_units}: Training complete, epochs: {len(history.history["loss"])}, min loss {np.min(history.history["val_loss"])}, total time {total_time} ----\n')

        # Plot training and validation loss
        print('\n---- Plotting loss ----\n')

        train_loss_l = np.array(history.history['loss'])
        valid_loss_l = np.array(history.history['val_loss'])

        plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
                    title='Encoder Train/Validation Loss',
                    figname=os.path.join(plots_dir, f'units{dense_units}_train_valid_loss.png'))

        # Plot some samples
        plot_sample(x_valid, samples=[0, 1, 2, 3], autoenc=autoenc,
                    figname=os.path.join(plots_dir, f'units{dense_units}_samples.png'))

        # save file with experiment configuration
        config_dict[f'{dense_units}'] = {
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
    min_train_loss = np.array([config_dict[str(d)]['min_train_loss'] for d in dense_layers])
    min_valid_loss = np.array([config_dict[str(d)]['min_valid_loss'] for d in dense_layers])
    fig = plt.figure(figsize=(8, 6))
    plt.plot(dense_layers, min_train_loss, label='Training')
    plt.plot(dense_layers, min_valid_loss, label='Validation')
    plt.xlabel('Dense units')
    plt.ylabel('Min loss')
    plt.legend()
    plt.savefig(os.path.join(plots_dir, f'min_loss_dense_units.png'), dpi=300)
