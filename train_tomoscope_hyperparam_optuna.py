# Train the ML model
import optuna
from optuna.trial import TrialState
from optuna.integration import TFKerasPruningCallback
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_slice, plot_contour

import argparse
from mlp_lhc_tomography.utils import sample_files
from local_models import Tomoscope
from local_utils import tomoscope_files_to_tensors, fast_tensor_load_encdec

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

# Train specific
train_cfg = {
    'epochs': 25, 'output_turns': 10,
    'cropping': [0, 0],
    'enc_dense_layers': [1024, 256],
    'dec_dense_layers': [],
    'enc_filters': [4, 8, 16],
    'dec_filters': [64, 32, 10],
    'enc_kernel_size': 5,
    'dec_kernel_size': [7, 7, 7],
    'enc_strides': [2, 2],
    'dec_strides': [2, 2],
    'enc_activation': 'relu',
    'dec_activation': 'relu',
    'final_activation': 'tanh',
    'enc_pooling': None, 'dec_pooling': None,
    'enc_pooling_size': [0, 0], 'dec_pooling_size': [0, 0],
    'enc_pooling_strides': [1, 1], 'dec_pooling_strides': [1, 1],
    'enc_pooling_padding': 'valid', 'dec_pooling_padding': 'valid',
    'enc_dropout': 0.0, 'dec_dropout': 0.0,
    'metrics': [], 'use_bias': False, 'batchnorm': False,
    'learning_rate': 1e-3,
    'dataset%': 0.5, 'loss': 'mse',
    'normalization': 'minmax', 'img_normalize': 'off',
    'ps_normalize': 'off',
    'batch_size': 32
}

param_space = {
    'enc_kernel_size': [
        '9,9,5', '9,7,5',
        '7,7,7', '7,7,5',
        '5,5,5', '3,3,3',
    ],
    'dec_kernel_size': [
        '9,9,5', '9,7,5',
        '7,7,7', '7,7,5',
        '5,5,5', '3,3,3',
    ],
    'enc_filters': ['4,8,16', '8,16,32', '16,32,64'],
    'dec_filters': ['64,32,10', '32,16,10', '4,8,10'],
    'enc_dense_layers': ['', '16', '64', '256'],
    'dec_dense_layers': ['', '16', '64', '256'],
}


def train_test_model(x_train, y_train, x_valid, y_valid, train_cfg, trial):
    """Train and test the model with the given hyperparameters.
    """
    hparams = {
        'enc_kernel_size': [int(i) for i in trial.suggest_categorical('e_kr_sz', param_space['enc_kernel_size']).split(',') if i != ''],
        'enc_filters': [int(i) for i in trial.suggest_categorical('e_flt', param_space['enc_filters']).split(',') if i != ''],
        'enc_dense_layers': [int(i) for i in trial.suggest_categorical('e_lrs', param_space['enc_dense_layers']).split(',') if i != ''],
        'dec_kernel_size': [int(i) for i in trial.suggest_categorical('d_kr_sz', param_space['dec_kernel_size']).split(',') if i != ''],
        'dec_filters': [int(i) for i in trial.suggest_categorical('d_flt', param_space['dec_filters']).split(',') if i != ''],
        'dec_dense_layers': [int(i) for i in trial.suggest_categorical('d_lrs', param_space['dec_dense_layers']).split(',') if i != ''],
    }
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
                   gc_after_trial=True, n_jobs=1, n_trials=2000, timeout=3600*1.5)
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

    # print(f'\n---- HyperParam tuning for tomoscope ----\n')
    # session_num = 0

    # # Load param space
    # # param_space = model_cfg.get(var_name, {})
    # keys, values = zip(*param_space.items())
    # total_runs = np.prod([len(v) for v in param_space.values()])

    # overall_dict = {}
    # for bundle in product(*values):
    #     hparams = dict(zip(keys, bundle))
    #     run_name = f"run-{session_num}"
    #     print(f'--- Starting trial: {run_name}/{total_runs}')
    #     print(hparams)
    #     start_t = time.time()
    #     history, loss = train_test_model(x_train, y_train, x_valid, y_valid,
    #                                      os.path.join(trial_dir, run_name), hparams)
    #     total_time = time.time() - start_t
    #     train_loss = np.min(history["loss"])
    #     valid_loss = np.min(history["val_loss"])
    #     overall_dict[run_name] = {
    #         'time': total_time, 'train': train_loss, 'valid': valid_loss, 'history': history, 'data%': train_cfg['dataset%']}
    #     overall_dict[run_name].update(hparams)
    #     print(
    #         f'---- Training complete, epochs: {len(history["loss"])}, train loss {np.min(history["loss"]):.2e}, valid loss {np.min(history["val_loss"]):.2e}, time {total_time:.2f}')

    #     session_num += 1

    #     # save every 10 sessions
    #     if session_num % 10 == 0:
    #         fname = os.path.join(hparams_dir, f'tomoscope_{timestamp}.pkl')
    #         with open(fname, 'wb') as handle:
    #             pickle.dump(overall_dict, handle,
    #                         protocol=pickle.HIGHEST_PROTOCOL)

    # # save final data
    # fname = os.path.join(hparams_dir, f'tomoscope_{timestamp}.pkl')
    # with open(fname, 'wb') as handle:
    #     pickle.dump(overall_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
