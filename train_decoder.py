# Train the ML model

from models import extendedCED
# from utils import load_model_data_new, normalize_params
from utils import plot_loss, decoder_files_to_tensors
import time
import glob
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

# Initialize parameters
data_dir = '/eos/user/k/kiliakis/tomo_data/datasets'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

# Data specific
IMG_OUTPUT_SIZE = 128
BUFFER_SIZE = 1000
BATCH_SIZE = 32  # 8
latent_dim = 7  # 6 + the new VrfSPS
additional_latent_dim = 1

# Train specific
train_cfg = {
    'decoder': {
        'epochs': 2,
        'lr': 2e-4,
    },
}

# Keep only a small percentage of the entire dataset
# for faster testing.
dataset_keep_percent = 0.1
# cnn_filters = [32, 64, 128, 256, 512, 1024]
cnn_filters = [32]

if __name__ == '__main__':

    args = parser.parse_args()
    # If input config file is provided, read input from config file
    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['train_cfg']
        cnn_filters = input_config['cnn_filters']
        dataset_keep_percent = input_config['dataset_keep_percent']
        timestamp = input_config['timestamp']

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
    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')
    assert os.path.exists(TRAINING_PATH)
    assert os.path.exists(VALIDATION_PATH)

    # create the directory to store the results
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=False)
    os.makedirs(plots_dir, exist_ok=False)

    # Create the datasets
    # First the training data
    files = glob.glob(TRAINING_PATH + '/*.pk')
    files = files[:int(len(files) * dataset_keep_percent)]

    # Shuffle them
    np.random.shuffle(files)
    # read input, divide in features/ label, create tensors
    x_train, y_train = decoder_files_to_tensors(files)

    # Then the validation data
    files = glob.glob(VALIDATION_PATH + '/*.pk')
    files = files[:int(len(files) * dataset_keep_percent)]

    # Shuffle them
    np.random.shuffle(files)
    # read input, divide in features/ label, create tensors
    x_valid, y_valid = decoder_files_to_tensors(files)

    # Model instantiation
    input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

    eCED = extendedCED(latent_dim, additional_latent_dim, input_shape,
                       filters=cnn_filters)

    print(eCED.decoder.summary())

    print('\n---- Training the decoder ----\n')

    optimizer = keras.optimizers.Adam(train_cfg['encoder']['lr'])
    eCED.decoder.compile(optimizer=optimizer, loss='mse')

    # callbacks, save the best model, and early stop if no improvement in val_loss
    stop_early = keras.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=5, restore_best_weights=True)
    save_best = keras.callbacks.ModelCheckpoint(filepath=os.path.join(weights_dir, 'decoder'),
                                                monitor='val_loss', save_best_only=True)

    start_time = time.time()
    history = eCED.decoder.fit(
        x_train, y_train, epochs=train_cfg['decoder']['epochs'],
        validation_data=(x_valid, y_valid), batch_size=BATCH_SIZE,
        callbacks=[stop_early, save_best])

    total_time = time.time() - start_time

    # Plot training and validation loss
    train_loss_l = np.array(history.history['loss'])
    valid_loss_l = np.array(history.history['val_loss'])

    plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
              title='decoder Train/Validation Loss',
              figname=os.path.join(plots_dir, 'decoder_train_valid_loss.png'))

    # Save the best model's weights
    # eCED.decoder = best_decoder
    # eCED.decoder.save_weights(os.path.join(
    #     weights_dir, 'eCED_weights_decoder.h5'), save_format='h5')

    # save file with experiment configuration
    config_dict = {}
    config_dict['decoder'] = {
        'epochs': train_cfg['decoder']['epochs'],
        'lr': train_cfg['decoder']['lr'],
        'dataset_percent': dataset_keep_percent,
        'cnn_filters': list(cnn_filters),
        'min_train_loss': float(np.min(train_loss_l)),
        'min_valid_loss': float(np.min(valid_loss_l)),
        'total_train_time': total_time,
        'used_gpus': len(gpus)
    }

    # save config_dict
    with open(os.path.join(trial_dir, 'decoder-summary.yml'), 'w') as configfile:
        yaml.dump(config_dict, configfile, default_flow_style=False)
