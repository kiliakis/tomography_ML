# Train the ML model

from models import extendedCED, mse_loss_encoder, mse_loss_decoder
from utils import load_model_data_new, unnormalize_params, assess_decoder
from utils import plot_loss
import time
import glob
from telnetlib import EC
import tensorflow as tf
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
BUFFER_SIZE = 50
BATCH_SIZE = 32  # 8
latent_dim = 7  # 6 + the new VrfSPS
additional_latent_dim = 1

# Train specific
models_to_train = ['encoder', 'decoder']
train_cfg = {
    'encoder': {
        'epochs': 2,
        'lr': 2e-4,
    },
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
    # TESTING_PATH = os.path.join(ML_dir, 'TESTING')
    assert os.path.exists(TRAINING_PATH)
    assert os.path.exists(VALIDATION_PATH)
    # assert os.path.exists(TESTING_PATH)
    
    # create the directory to store the results 
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=False)
    os.makedirs(plots_dir, exist_ok=False)

    # Create the datasets
    files = glob.glob(TRAINING_PATH + '/*.pk')
    files = files[:int(len(files) * dataset_keep_percent)]
    # train_dataset = tf.data.Dataset.list_files(TRAINING_PATH + '/*.pk')
    train_dataset = tf.data.Dataset.from_tensor_slices(files)
    train_dataset = train_dataset.map(lambda x: tf.py_function(load_model_data_new, [x],
                                                               [tf.float32, tf.float32, tf.float32, tf.string,
                                                                tf.float32, tf.float32, tf.float32, tf.float32,
                                                                tf.float32, tf.float32, tf.float32, tf.float32,
                                                                tf.float32]))

    train_dataset = train_dataset.shuffle(BUFFER_SIZE)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    files = glob.glob(VALIDATION_PATH + '/*.pk')
    files = files[:int(len(files) * dataset_keep_percent)]
    # valid_dataset = tf.data.Dataset.list_files(VALIDATION_PATH + '/*.pk')
    valid_dataset = tf.data.Dataset.from_tensor_slices(files)
    valid_dataset = valid_dataset.map(lambda x: tf.py_function(load_model_data_new, [x],
                                                               [tf.float32, tf.float32, tf.float32, tf.string,
                                                                tf.float32, tf.float32, tf.float32, tf.float32,
                                                                tf.float32, tf.float32, tf.float32, tf.float32,
                                                                tf.float32]))

    valid_dataset = valid_dataset.shuffle(BUFFER_SIZE)
    valid_dataset = valid_dataset.batch(BATCH_SIZE)

    # Model instantiation
    input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

    eCED = extendedCED(latent_dim, additional_latent_dim, input_shape,
                       filters=cnn_filters)

    print(eCED.encoder.summary())
    print(eCED.extender.summary())
    print(eCED.decoder.summary())
    config_dict = {}

    if 'encoder' in models_to_train:
 
        # Train the encoder
        print('\n---- Training the encoder ----\n')
        optimizer = tf.keras.optimizers.Adam(train_cfg['encoder']['lr'])
        train_loss_l, valid_loss_l = [], []
        best_encoder = None
        total_time = 0
        for epoch in range(train_cfg['encoder']['epochs']):
            start_time = time.time()
            l_encs = []
            # Iterate over each batch
            for n, (norm_turns, T_imgs, PS_imgs, fns, phErs, enErs, bls, intens,
                    Vrfs, mus, VrfSPSs, _, _) in train_dataset.enumerate():
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()

                # Calculate loss and gradients
                with tf.GradientTape() as tape:
                    l_enc = mse_loss_encoder(eCED, T_imgs, phErs, enErs, bls,
                                             intens, Vrfs, mus, VrfSPSs)
                    l_encs.append(l_enc.numpy())
                    gradients = tape.gradient(l_enc,
                                              eCED.encoder.trainable_variables)
                # Apply the gradients
                optimizer.apply_gradients(
                    zip(gradients, eCED.encoder.trainable_variables))
            # Record the average train loss for the entire epoch
            train_loss_l.append(np.mean(np.array(l_encs)))
            print()
            end_time = time.time()
            total_time += end_time - start_time

            # Repeat for validation data, do not update the gradients
            loss = tf.keras.metrics.Mean()
            l_encs = []
            for n, (norm_turns, T_imgs, PS_imgs, fns,  phErs, enErs, bls, intens,
                    Vrfs, mus, VrfSPSs, _, _) in valid_dataset.enumerate():
                l_enc = mse_loss_encoder(eCED, T_imgs, phErs, enErs, bls, intens,
                                         Vrfs, mus, VrfSPSs)
                loss(l_enc)
                l_encs.append(l_enc.numpy())

            # Record the average validation loss
            valid_loss_l.append(np.mean(np.array(l_encs)))

            # Save if validation loss is minimum
            if valid_loss_l[-1] == np.min(valid_loss_l):
                best_encoder = eCED.encoder

            elbo = loss.result()
            # display.clear_output(wait=False)
            print('Epoch: {}, Train set loss: {:.4f}, Valid set loss: {:.4f}, time elapse: {:.4f}'.format(
                epoch, train_loss_l[-1], valid_loss_l[-1], end_time - start_time))
            # # TODO: assess encoder visually
            #assess_model(model, norm_turns[0:1], T_imgs[0:1], PS_imgs[0:1], epoch)
            # valid_pred_pars = eCED.encode(T_imgs[0:1])

            # validDelPar = np.array(unnormalize_params(*list(valid_pred_pars[0].numpy()))) -\
            #     np.array([phErs[0].numpy(), enErs[0].numpy(), bls[0].numpy(),
            #               intens[0].numpy(), Vrfs[0].numpy(), mus[0].numpy(),
            #               VrfSPSs[0].numpy()])
            # plt.figure()
            # plt.plot(validDelPar/np.array([1, 1, 1e-11, 1e10, 0.1, 0.1, 0.1]), 's')

        # Plot training and validation loss
        train_loss_l = np.array(train_loss_l)
        valid_loss_l = np.array(valid_loss_l)

        plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
                  title='Encoder Train/Validation Loss',
                  figname=os.path.join(plots_dir, 'encoder_train_valid_loss.png'))

        # Save the best model's weights
        eCED.encoder = best_encoder
        eCED.encoder.save_weights(os.path.join(
            weights_dir, 'eCED_weights_encoder.h5'), save_format='h5')

        # save file with experiment configuration
        config_dict['encoder'] = {
            'epochs': train_cfg['encoder']['epochs'],
            'lr': train_cfg['encoder']['lr'],
            'dataset_percent': dataset_keep_percent,
            'cnn_filters': list(cnn_filters),
            'min_train_loss': float(np.min(train_loss_l)),
            'min_valid_loss': float(np.min(valid_loss_l)),
            'total_train_time': total_time,
            'used_gpus': len(gpus)
        }

    if 'decoder' in models_to_train:
        print('\n---- Training the decoder ----\n')

        # Train the extender + decoder
        optimizer = tf.keras.optimizers.Adam(train_cfg['decoder']['lr'])
        train_loss_l, valid_loss_l = [], []
        best_extender = None
        best_decoder = None
        total_time = 0

        for epoch in range(train_cfg['decoder']['epochs']):
            start_time = time.time()
            l_imgs_training = []
            # Iterate over each batch
            for n, (norm_turns, T_imgs, PS_imgs, fns, phErs, enErs, bls, intens,
                    Vrfs, mus, VrfSPSs, _, _) in train_dataset.enumerate():
                print('.', end='')
                if (n+1) % 100 == 0:
                    print()
                # l_img, l_lat = train_step_extended_physical(model, norm_turns, T_imgs, PS_imgs,  phErs, enErs, bls, intens, optimizer)
                norm_turns = tf.expand_dims(norm_turns, axis=0)
                # Calculate loss and gradients
                with tf.GradientTape() as tape:
                    l_img = mse_loss_decoder(eCED, norm_turns, PS_imgs, phErs,
                                             enErs, bls, intens, Vrfs, mus, VrfSPSs)
                    l_imgs_training.append(l_img.numpy())
                    gradients = tape.gradient(
                        l_img, eCED.extender.trainable_variables + eCED.decoder.trainable_variables)

                # Apply the gradients
                optimizer.apply_gradients(zip(
                    gradients, eCED.extender.trainable_variables + eCED.decoder.trainable_variables))
            train_loss_l.append(np.mean(np.array(l_imgs_training)))
            print()
            end_time = time.time()
            total_time += end_time - start_time

            # Repeat for validation data, do not update the gradients
            loss = tf.keras.metrics.Mean()
            l_imgs_validation = []
            for n, (norm_turns, T_imgs, PS_imgs, fns,  phErs, enErs, bls, intens,
                    Vrfs, mus, VrfSPSs, _, _) in valid_dataset.enumerate():
                norm_turns = tf.expand_dims(norm_turns, axis=0)
                l_img = mse_loss_decoder(eCED, norm_turns, PS_imgs, phErs, enErs,
                                         bls, intens, Vrfs, mus, VrfSPSs)
                loss(l_img)
                l_imgs_validation.append(l_img.numpy())

            # Record validation loss
            valid_loss_l.append(np.mean(np.array(l_imgs_validation)))
            # Save if validation loss is minimum
            if valid_loss_l[-1] == np.min(valid_loss_l):
                best_extender = eCED.extender
                best_decoder = eCED.decoder
            elbo = loss.result()
            # display.clear_output(wait=False)
            print('Epoch: {}, Train set loss: {:.4f}, Valid set loss: {:.4f}, time elapse: {:.4f}'.format(
                epoch, train_loss_l[-1], valid_loss_l[-1], end_time - start_time))
            assess_decoder(eCED, norm_turns[0][0:1], PS_imgs[0:1], phErs[0],
                           enErs[0], bls[0], intens[0], Vrfs[0], mus[0],
                           VrfSPSs[0], epoch, figname=os.path.join(plots_dir, 'assess_decoder.png'),
                           savefig=True)

        # Plot training and validation loss
        train_loss_l = np.array(train_loss_l)
        valid_loss_l = np.array(valid_loss_l)

        plot_loss({'Training': train_loss_l, 'Validation': valid_loss_l},
                  title='Decoder Train/Validation Loss',
                  figname=os.path.join(plots_dir, 'decoder_train_valid_loss.png'))

        # Save the best model's weights
        eCED.extender = best_extender
        eCED.decoder = best_decoder
        eCED.extender.save_weights(os.path.join(
            weights_dir, 'eCED_weights_extender.h5'), save_format='h5')
        eCED.decoder.save_weights(os.path.join(
            weights_dir, 'eCED_weights_decoder.h5'), save_format='h5')

        # save file with experiment configuration
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
    if len(config_dict):
        with open(os.path.join(trial_dir, 'summary.yml'), 'w') as configfile:
            yaml.dump(config_dict, configfile, default_flow_style=False)
