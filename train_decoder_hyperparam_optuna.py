# Train the ML model
import optuna
from optuna.trial import TrialState
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice, plot_contour

import argparse
from mlp_lhc_tomography.utils import sample_files, decoder_files_to_tensors
from mlp_lhc_tomography.utils import fast_tensor_load
from mlp_lhc_tomography.models import Decoder
import pickle
import time
import tensorflow as tf
# from tensorboard.plugins.hparams import api as hp
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
data_dir = './tomo_data/datasets_decoder_TF_24-03-23'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

DATA_LOAD_METHOD='FAST_TENSOR'

var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']

N_TRIALS=1000
TIMEOUT=60*60*3 # 2 hours

# Train specific
train_cfg = {
    'epochs': 20,
    'dense_layers': [len(var_names) + 1, 256, 1024],
    'filters': [32, 16, 8, 1],
    'kernel_size': 9,
    'strides': [2, 2],
    'final_kernel_size': 5,
    'activation': 'relu',
    'final_activation': 'tanh',
    'dropout': 0.,
    'loss': 'mse',
    'lr': 1e-3,
    'dataset%': 0.1,
    'normalization': 'minmax',
    'ps_normalize': 'off',
    'batch_size': 32
}

param_space = {
    'kernel_size': [3, 5, 7, 9, 11, 13],
    'final_kernel_size': [3, 5, 7, 9],
    'filters': [
                '32,16,8,1', '32,8,1',
                '32,16,1', '64,16,8,1',
                '16,8,1', '16,8,4,1'
                ],
    'dense_layers': ['8,64,1024', '8,256,1024',
                     '8,256', '8,1024','8,512',
                     '8']
}


def train_test_model(x_train, y_train, x_valid, y_valid, trial):
    hparams = {
        'filters': [int(i) for i in trial.suggest_categorical('flt', param_space['filters']).split(',') if i != ''],
        # 'kernel_size': [int(i) for i in trial.suggest_categorical('kr_sz', param_space['kernel_size']).split(',') if i != ''],
        'kernel_size': trial.suggest_categorical('kr_sz', param_space['kernel_size']),
        'final_kernel_size': trial.suggest_categorical('fnl_kr_sz', param_space['final_kernel_size']),
        'dense_layers': [int(i) for i in trial.suggest_categorical('lrs', param_space['dense_layers']).split(',') if i != ''],
    }
    cfg = train_cfg.copy()
    cfg.update(hparams)

    output_shape = y_train.shape[1:]
    model = Decoder(output_shape, **cfg)
    model.model.fit(
        x=x_train, y=y_train,
        epochs=cfg['epochs'],
        validation_data=(x_valid, y_valid),
        callbacks=[TFKerasPruningCallback(trial, "val_loss")],
        batch_size=cfg['batch_size'],
        verbose=0)
    val_loss = model.model.evaluate(x_valid, y_valid)

    return val_loss


if __name__ == '__main__':
    args = parser.parse_args()

    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['decoder']
        if 'param_space' in input_config:
            param_space = input_config['param_space']
        timestamp = input_config['timestamp']
    
    print('Configuration:')
    for k, v in train_cfg.items():
        print(k, v)

    print('Param space:')
    for param, values in param_space.items():
        print(param, values)

    # Initialize directories
    trial_dir = os.path.join('./hparam_trials/', timestamp)

    weights_dir = os.path.join(trial_dir)
    plots_dir = os.path.join(trial_dir)
    logs_dir = os.path.join(trial_dir)
    hparams_dir = os.path.join(trial_dir)

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
    # Create the datasets
    if DATA_LOAD_METHOD=='TENSOR':
        # First the training data
        file_names = sample_files(
            TRAINING_PATH, train_cfg['dataset%'], keep_every=1)
        print('Number of Training files: ', len(file_names))
        x_train, y_train = decoder_files_to_tensors(
            file_names, normalization=train_cfg['normalization'],
            ps_normalize=train_cfg['ps_normalize'])

        # Repeat for validation data
        file_names = sample_files(
            VALIDATION_PATH, train_cfg['dataset%'], keep_every=1)
        print('Number of Validation files: ', len(file_names))

        x_valid, y_valid = decoder_files_to_tensors(
            file_names, normalization=train_cfg['normalization'],
            ps_normalize=train_cfg['ps_normalize'])
        
        # drop column from y_train, y_valid
        x_train = tf.concat([tf.expand_dims(tf.gather(x_train, i, axis=1), axis=1)
                            for i in train_cfg['loss_weights']], -1)
        print('x_train shape: ', x_train.shape)

        x_valid = tf.concat([tf.expand_dims(tf.gather(x_valid, i, axis=1), axis=1)
                            for i in train_cfg['loss_weights']], -1)
        print('x_valid shape: ', x_valid.shape)
    
    elif DATA_LOAD_METHOD == 'FAST_TENSOR':
        assert train_cfg['normalization'] == 'minmax'
        assert train_cfg['ps_normalize'] == 'off'

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

    study = optuna.create_study(study_name=f'decoder_{timestamp}',
        direction='minimize', pruner=optuna.pruners.MedianPruner())
    try:
        study.optimize(lambda trial: train_test_model(x_train, y_train, x_valid, y_valid, trial),
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
    fname = os.path.join(hparams_dir, f'decoder_{timestamp}.csv')
    all_trials.to_csv(fname)

    # Save some plots
    fig = plot_optimization_history(study)
    fig.write_image(os.path.join(plots_dir, f'decoder_optimization_history.png'))

    fig = plot_parallel_coordinate(study)
    fig.write_image(os.path.join(plots_dir, f'decoder_parallel_coordinate.png'))

    fig = plot_param_importances(study)
    fig.write_image(os.path.join(plots_dir, f'decoder_param_importances.png'))

    fig = plot_slice(study)
    fig.write_image(os.path.join(plots_dir, f'decoder_slice.png'))

    fig = plot_contour(study)
    fig.write_image(os.path.join(plots_dir, f'decoder_contour.png'))

    # save the study object
    fname = os.path.join(trial_dir, f'decoder_{timestamp}_study.pkl')
    with open(fname, 'wb') as handle:
        pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
