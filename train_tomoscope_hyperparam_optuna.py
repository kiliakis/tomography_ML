# Train the ML model
import optuna
from optuna.trial import TrialState
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice, plot_contour

import argparse
from utils import sample_files
from models import Tomoscope
from utils import tomoscope_files_to_tensors, fast_tensor_load_encdec

from itertools import product
import pickle
import time
import tensorflow as tf
import yaml
import os
import numpy as np
from datetime import datetime
import matplotlib as mpl
mpl.use('Agg')


parser = argparse.ArgumentParser(description='Train the encoder/ decoder models',
                                 usage='python train_model.py -c config.yml')

parser.add_argument('-c', '--config', type=str, default=None,
                    help='A yaml configuration file with all training parameters.')

# Initialize parameters
# data_dir = '/eos/user/k/kiliakis/tomo_data/datasets_encoder_02-12-22'
data_dir = './tomo_data/datasets_tomoscope_TF_24-03-23'

timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

DATA_LOAD_METHOD = 'FAST_TENSOR'  # it can be TENSOR or DATASET or FAST_TENSOR

N_TRIALS = 100
TIMEOUT = 60*60*1  # 12 hours


# Train specific
train_cfg = {
    'epochs': 100, 'output_turns': 1,
    'cropping': [0, 0],
    'enc_filters': [16, 32, 64, 128],
    'dec_filters': [64, 32, 16],
    'enc_kernel_size': 4,
    'dec_kernel_size': 4,
    'enc_strides': [2, 2],
    'dec_strides': [2, 2],
    'enc_activation': 'relu',
    'dec_activation': 'relu',
    'final_activation': 'tanh',
    'enc_batchnorm': False, 'dec_batchnorm': True,
    'enc_conv_padding': 'same', 'dec_conv_padding': 'same',
    'enc_dropout': 0.0, 'dec_dropout': 0.0,
    'metrics': [], 'use_bias': False, 'batchnorm': False,
    'learning_rate': 2e-4,
    'dataset%': 1,
    'normalization': 'minmax', 'img_normalize': 'off',
    'ps_normalize': 'off',
    'batch_size': 32
}

# # Train specific
# train_cfg = {
#     'epochs': 100, 'output_turns': 1,
#     'cropping': [0, 0],
#     'enc_dense_layers': [],
#     'dec_dense_layers': [],
#     'enc_filters': [16, 32, 64],
#     'dec_filters': [64, 32, 1],
#     'enc_kernel_size': [9, 7, 5],
#     'dec_kernel_size': [7, 7, 7],
#     'enc_strides': [2, 2],
#     'dec_strides': [2, 2],
#     'enc_activation': 'relu',
#     'dec_activation': 'relu',
#     'final_activation': 'tanh',
#     'enc_pooling': None, 'dec_pooling': None,
#     'enc_pooling_size': [0, 0], 'dec_pooling_size': [0, 0],
#     'enc_pooling_strides': [1, 1], 'dec_pooling_strides': [1, 1],
#     'enc_pooling_padding': 'valid', 'dec_pooling_padding': 'valid',
#     'enc_dropout': 0.0, 'dec_dropout': 0.0,
#     'metrics': [], 'use_bias': False, 'batchnorm': False,
#     'learning_rate': 1e-3,
#     'dataset%': 1, 'loss': 'mse',
#     'normalization': 'minmax', 'img_normalize': 'off',
#     'ps_normalize': 'off',
#     'batch_size': 32
# }

param_space = {
    'enc_filters': ['16,32,64,128', '8,16,32,64', '32,64,128,256', '16,32,64,64'],
    'dec_filters': ['64,32,16', '32,16,8', '128,64,32', '64,64,32', '64,32,32'],
    'final_activation': ['linear', 'tanh'],
    'enc_activation': ['relu', 'leakyrelu'],
    'dec_activation': ['relu', 'leakyrelu'],
    'enc_batchnorm': [False, True],
    'dec_batchnorm': [False, True]
}

split_keys = ['enc_kernel_size', 'enc_filters', 'enc_dense_layers', 'dec_kernel_size', 'dec_filters', 'dec_dense_layers',
              'cropping']


category_keys = {
    'enc_kernel_size': 'e_kr_sz',
    'enc_filters': 'e_flt',
    'enc_dense_layers': 'e_lrs',
    'dec_kernel_size': 'd_kr_sz',
    'dec_filters': 'd_flt',
    'dec_dense_layers': 'd_lrs',
    'cropping': 'crp',
    'use_bias': 'bias',
    'batch_size': 'btch_sz',
    'final_activation': 'fnl_activ',
    'enc_activation': 'enc_activ',
    'dec_activation': 'dec_activ',
    'enc_batchnorm': 'enc_norm',
    'dec_batchnorm': 'dec_norm'

}



def train_test_model(x_train, y_train, x_valid, y_valid, train_cfg, trial):
    """Train and test the model with the given hyperparameters.
    """
    hparams = {}
    for var in param_space:
        value = trial.suggest_categorical(
            category_keys.get(var, var), param_space[var])
        if var in split_keys and isinstance(value, str):
            value = [int(i.strip()) for i in value.split(',') if i != '']
        hparams[var] = value

    # hparams = {
    #     'enc_kernel_size': [int(i) for i in trial.suggest_categorical('e_kr_sz', param_space['enc_kernel_size']).split(',') if i != ''],
    #     'enc_filters': [int(i) for i in trial.suggest_categorical('e_flt', param_space['enc_filters']).split(',') if i != ''],
    #     'enc_dense_layers': [int(i) for i in trial.suggest_categorical('e_lrs', param_space['enc_dense_layers']).split(',') if i != ''],
    #     'dec_kernel_size': [int(i) for i in trial.suggest_categorical('d_kr_sz', param_space['dec_kernel_size']).split(',') if i != ''],
    #     'dec_filters': [int(i) for i in trial.suggest_categorical('d_flt', param_space['dec_filters']).split(',') if i != ''],
    #     'dec_dense_layers': [int(i) for i in trial.suggest_categorical('d_lrs', param_space['dec_dense_layers']).split(',') if i != ''],
    # }
    cfg = train_cfg.copy()
    cfg.update(hparams)

    input_shape = x_train.shape[1:]

    model = Tomoscope(input_shape=input_shape, **cfg)

    # start_t = time.time()
    model.model.fit(
        x=x_train, y=y_train,
        epochs=cfg['epochs'],
        validation_data=(x_valid, y_valid),
        callbacks=[TFKerasPruningCallback(trial, "val_loss")],
        batch_size=cfg['batch_size'],
        verbose=0)
    # total_t = time.time() - start_t
    val_loss = model.model.evaluate(x_valid, y_valid)

    del model
    return val_loss


if __name__ == '__main__':
    args = parser.parse_args()

    # If input config file is provided, read input from config file
    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['tomoscope']
        if 'param_space' in train_cfg['tomoscope']:
            param_space = input_config['tomoscope']['param_space']
        timestamp = input_config['timestamp']

    print('Configuration:')
    for k, v in train_cfg.items():
        print(k, v)

    # Initialize directories
    # trial_dir = './'
    trial_dir = os.path.join('./hparam_trials/', timestamp)
    weights_dir = os.path.join(trial_dir, 'weights')
    plots_dir = os.path.join(trial_dir, 'plots')
    logs_dir = os.path.join(trial_dir, 'logs')
    hparams_dir = os.path.join(trial_dir, 'hparams')

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
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(hparams_dir, exist_ok=True)

    np.random.seed(0)

    start_t = time.time()
    if DATA_LOAD_METHOD == 'TENSOR':
        # Create the datasets
        # 1. Randomly select the training data
        file_names = sample_files(
            TRAINING_PATH, train_cfg['dataset%'], keep_every=1)
        print('Number of Training files: ', len(file_names))
        x_train, turn_train, latent_train, y_train = tomoscope_files_to_tensors(
            file_names, normalization=train_cfg['normalization'],
            img_normalize=train_cfg['img_normalize'],
            ps_normalize=train_cfg['ps_normalize'], num_turns=train_cfg['output_turns'])

        # Repeat for validation data
        file_names = sample_files(
            VALIDATION_PATH, train_cfg['dataset%'], keep_every=1)
        print('Number of Validation files: ', len(file_names))

        x_valid, turn_valid, latent_valid, y_valid = tomoscope_files_to_tensors(
            file_names, normalization=train_cfg['normalization'],
            img_normalize=train_cfg['img_normalize'],
            ps_normalize=train_cfg['ps_normalize'], num_turns=train_cfg['output_turns'])
    elif DATA_LOAD_METHOD == 'FAST_TENSOR':
        assert train_cfg['normalization'] == 'minmax'
        assert train_cfg['img_normalize'] == 'off'
        assert train_cfg['ps_normalize'] == 'off'

        TRAINING_PATH = os.path.join(ML_dir, 'tomoscope-training-??.npz')
        VALIDATION_PATH = os.path.join(ML_dir, 'tomoscope-validation-??.npz')

        x_train, turn_train, latent_train, y_train = fast_tensor_load_encdec(
            TRAINING_PATH, train_cfg['dataset%'])
        print('Number of Training files: ', len(y_train))

        x_valid, turn_valid, latent_valid, y_valid = fast_tensor_load_encdec(
            VALIDATION_PATH, train_cfg['dataset%'])
        print('Number of Validation files: ', len(y_valid))

        y_train = y_train[:, :, :, :train_cfg['output_turns']]
        y_valid = y_valid[:, :, :, :train_cfg['output_turns']]
    else:
        exit('DATA_LOAD_METHOD not recognised')

    print(
        f'\n---- Input files have been read, elapsed: {time.time() - start_t} ----\n')

    # Model instantiation
    start_t = time.time()

    study = optuna.create_study(study_name=f'tomoscope_{timestamp}',
        direction='minimize', pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(lambda trial: train_test_model(x_train, y_train, x_valid, y_valid, train_cfg, trial),
                   gc_after_trial=True, n_jobs=1, n_trials=N_TRIALS, timeout=TIMEOUT)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))
    
    all_trials = study.trials_dataframe()
    fname = os.path.join(hparams_dir, f'tomoscope_{timestamp}.csv')
    all_trials.to_csv(fname)

    # Save some plots
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(plots_dir, 'optimization_history.png'))

    fig = plot_parallel_coordinate(study)
    fig.write_image(os.path.join(plots_dir, 'parallel_coordinate.png'))

    fig = plot_param_importances(study)
    fig.write_image(os.path.join(plots_dir, 'param_importances.png'))

    fig = plot_slice(study)
    fig.write_image(os.path.join(plots_dir, 'slice.png'))

    fig = plot_contour(study)
    fig.write_image(os.path.join(plots_dir, 'contour.png'))

    # save the study object
    fname = os.path.join(trial_dir, f'tomoscope_{timestamp}_study.pkl')
    with open(fname, 'wb') as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)

