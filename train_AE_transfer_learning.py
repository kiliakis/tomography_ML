# Train the ML model
from utils import sample_files, load_encoder_data, plot_loss, plot_multi_loss
from utils import fast_tensor_load, get_model_size, plot_feature_extractor_evaluation
from utils import plot_sample, visualize_weights

from models import FeatureExtractor, AutoEncoderTranspose, AutoEncoderSkipAhead, VariationalAutoEncoder

import time
import tensorflow as tf
from tensorflow import keras
import yaml
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

import argparse

parser = argparse.ArgumentParser(description='Train the encoder/ decoder models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

IMG_OUTPUT_SIZE = 128
DATA_LOAD_METHOD = 'FAST_TENSOR'  # it can be TENSOR or DATASET or FAST_TENSOR

num_Turns_Case = 1
var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']
# Initialize parameters
data_dir = './tomo_data/datasets_encoder_TF_08-11-23'

# data_dir = './tomo_data/datasets'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
timestamp = 'testing'
print('Using timestamp: ', timestamp)

# load_autoencoder = 'trials/2023_11_13_10-40-21/weights/autoenc_dense_1024_16.h5'
# load_autoencoder = 'trials/testing/weights/autoenc_dense.h5'
load_autoencoder = ''

# Train specific
train_cfg = {
    'dataset%': 1,
    'normalization': 'minmax',
    'img_normalize': 'off',
    'batch_size': 128
}

autoenc_cfg = {
    'epochs': 100,
    'filters': [8, 16, 32],
    'dense_layers': [1024, 64],
    'decoder_dense_layers': [1024],
    'cropping': [14, 14],
    'kernel_size': 5,
    'strides': [2, 2],
    'conv_activation': 'relu',
    'enc_activation': 'relu',
    'dec_activation': 'relu',
    'alpha': 0.1,
    'final_activation': 'linear',
    'conv_padding': 'same',
    'pooling': None,
    'pooling_size': [0, 0],
    'pooling_strides': [1, 1],
    'pooling_padding': 'valid',
    'dropout': 0.0,
    'loss': 'mae',
    'lr': 1e-4,
    'use_bias': False,
    'conv_batchnorm': False,
    'dense_batchnorm': False,
}

feature_extractor_cfg = {
    'epochs': 100,
    # 'dense_layers': [1024, 256, 64],
    'dense_layers': [256, 1024, 256, 64],
    'activation': 'relu',
    'dropout': 0.0,
    'loss': 'mae',
    'lr': 1e-4,
    'use_bias': True,
    'batchnorm': False,
}


if __name__ == '__main__':
    args = parser.parse_args()
    # If input config file is provided, read input from config file
    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['train_cfg']
        autoenc_cfg = input_config['autoenc_cfg']
        feature_extractor_cfg = input_config['feature_extractor_cfg']
        timestamp = input_config['timestamp']
        load_autoencoder = input_config.get('load_autoencoder', load_autoencoder)

    print('Train Configuration:')
    for k, v in train_cfg.items():
        print(k, v)

    print('Autoencoder Configuration:')
    for k, v in autoenc_cfg.items():
        print(k, v)

    print('Feature extractor Configuration:')
    for k, v in feature_extractor_cfg.items():
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

    np.random.seed(42)

    start_t = time.time()

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

    end_t = time.time()
    print(
        f'\n---- Input files have been read, elapsed: {end_t - start_t} ----\n')

    config_dict = {}
    config_dict['timestamp'] = timestamp
    config_dict['train_cfg'] = train_cfg
    if load_autoencoder:
        config_dict['autoenc_cfg'] = load_autoencoder
    else:
        config_dict['autoenc_cfg'] = autoenc_cfg
    config_dict['feature_extractor_cfg'] = feature_extractor_cfg


    # Autoencoder Model instantiation
    start_t = time.time()

    if load_autoencoder:
        autoenc = keras.models.load_model(load_autoencoder)
        encoder = keras.Model(autoenc.inputs, 
                              outputs=autoenc.get_layer('LatentSpace').output)
        print(
            f'\n---- Autonecoder loaded, size: {get_model_size(autoenc):.1f}MB elapsed: {time.time() - start_t} ----\n')
    else:
        autoenc = AutoEncoderTranspose(
            input_shape=x_train.shape[1:], **autoenc_cfg)
        encoder = autoenc.encoder
        print(autoenc.model.summary())
        # print(encoder.summary())
        print(
            f'\n---- Models initialized, size: {get_model_size(autoenc.model):.1f}MB elapsed: {time.time() - start_t} ----\n')

        # tf.config.run_functions_eagerly(True)
        # callbacks, save the best model, and early stop if no improvement in val_loss
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    patience=10, restore_best_weights=False)
        save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, f'autoenc_dense.h5'),
                                                    monitor='val_loss', mode='min', save_best_only=True)

        start_time = time.time()

        history = autoenc.model.fit(
            x=x_train, y=x_train, 
            validation_data=(x_valid, x_valid),
            # x=y_train, y=x_train, 
            # validation_data=(y_valid, x_valid),
            epochs=autoenc_cfg['epochs'],
            callbacks=[stop_early, save_best],
            batch_size=train_cfg['batch_size'],
            verbose=1)

        # Load the best weights
        autoenc.load(os.path.join(weights_dir, f'autoenc_dense.h5'))

        total_time = time.time() - start_time
        print(
                f'\n---- Training complete, epochs: {len(history.history["loss"])}, min loss {np.min(history.history["val_loss"])}, total time {total_time} ----\n')

        # Plot training and validation loss
        print('\n---- Plotting loss ----\n')

        train_loss_l = np.array(history.history['loss'])
        valid_loss_l = np.array(history.history['val_loss'])

        plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
                    title='Autoencoder Train/Validation Loss',
                    figname=os.path.join(plots_dir, f'autoenc_train_valid_loss.png'))

        plot_sample(x_valid, samples=[0, 1, 2, 3], autoenc=autoenc,
                    figname=os.path.join(plots_dir, f'autoenc_samples.png'))

        # save file with experiment configuration
        config_dict['autoenc'] = {
            'epochs': len(history.history["loss"]),
            'min_train_loss': float(np.min(train_loss_l)),
            'min_valid_loss': float(np.min(valid_loss_l)),
            'total_train_time': total_time,
            'used_gpus': len(gpus)
        }

        # Visualize the weights
        visualize_weights(timestamp, f'autoenc_dense.h5')
    
    # Encode the train samples to get the latent space
    x_latent_train = tf.concat([encoder.predict(x_train), tf.expand_dims(y_train[:, 3], axis=1)], axis=1)
    x_latent_valid = tf.concat([encoder.predict(x_valid), tf.expand_dims(y_valid[:, 3], axis=1)], axis=1)
    
    # print the shape of the latent space
    print(f'Latent space shape: {x_latent_train.shape}')

    # Boxplot of the latent space
    # fig = plt.figure(figsize=(8, 6))
    # bxplots = plt.boxplot(x_latent_train)
    # plt.xticks(rotation=90)
    # plt.xlabel('Latent space')
    # plt.ylabel('Value')
    # plt.savefig(os.path.join(plots_dir, f'latent_space_boxplot.png'), dpi=300)
    # plt.close()


    model_cfg = {}
    models = {}
    for var_name in var_names:
        start_t = time.time()
        i = var_names.index(var_name)
        print(f'\n---- Initializing model: {var_name} ----')

        # cfg = train_cfg.copy()
        # cfg.update(model_cfg.get(var_name, {}))
        model_cfg[var_name] = feature_extractor_cfg.copy()
        model = FeatureExtractor(input_shape=x_latent_train.shape[1:],
                                output_features=1,
                                output_name=var_name, **model_cfg[var_name])

        # print(model.model.summary())

        if 'TENSOR' in DATA_LOAD_METHOD:
            models[var_name] = {'model': model.model,
                                'train': tf.gather(y_train, i, axis=1),
                                'valid': tf.gather(y_valid, i, axis=1)}

        elif DATA_LOAD_METHOD == 'DATASET':
            # method not supported
            pass

        print(
            f'---- Model initialized, size: {get_model_size(model.model):.2f}MB, elapsed: {time.time() - start_t} ----\n')

    historyMulti = {}
    latent_pred = np.empty(y_valid.shape)
    for i, var_name in enumerate(var_names):
        model = models[var_name]['model']

        # Train the encoder
        print(f'\n---- {var_name}: Training the encoder ----')

        start_time = time.time()

        # callbacks, save the best model, and early stop if no improvement in val_loss
        stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                patience=5, restore_best_weights=True)
        save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, f'feature_extractor_{var_name}.h5'),
                                                    monitor='val_loss', save_best_only=True)
        if 'TENSOR' in DATA_LOAD_METHOD:
            history = model.fit(
                x=x_latent_train, y=models[var_name]['train'],
                epochs=model_cfg[var_name]['epochs'],
                validation_data=(x_latent_valid, models[var_name]['valid']),
                callbacks=[save_best],
                batch_size=train_cfg['batch_size'],
                verbose=0)
        elif DATA_LOAD_METHOD == 'DATASET':
            # method not supported
            pass
        historyMulti[f'{var_name}_loss'] = history.history['loss']
        historyMulti[f'{var_name}_val_loss'] = history.history['val_loss']

        total_time = time.time() - start_time
        print(
            f'---- {var_name}: Training complete, epochs: {len(history.history["loss"])}, min loss {np.min(history.history["val_loss"])}, total time {total_time} ----\n')

        latent_pred[:, i] = model.predict(x_latent_valid)[:,0]


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
    # print(train_loss_l)
    # print(valid_loss_l)

    plot_loss({'training': train_loss_l, 'validation': valid_loss_l},
            title='Feature Extractor Train/Validation Loss',
              figname=os.path.join(plots_dir, 'feature_extractor_train_valid_loss.png'))

    plot_multi_loss(historyMulti, title='Feature Extractor loss per output',
            figname=os.path.join(plots_dir, 'feature_extractor_per_output_loss.png'))

    # Plot the evaluation of the feature extractor
    print('\n---- Plotting feature extractor evaluation ----\n')
    plot_feature_extractor_evaluation(latent_pred=latent_pred, latent_true=y_valid, 
                                      normalization=train_cfg['normalization'],
                                      figname=os.path.join(plots_dir, 'feature_extractor_evaluation.png'))


    config_dict['feature_extractor'] = {'losses': {}}
    for k, v in historyMulti.items():
        config_dict['feature_extractor']['losses'][k] = float(np.min(v))

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
