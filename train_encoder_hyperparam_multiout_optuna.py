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
from utils import encoder_files_to_tensors, fast_tensor_load
from models import EncoderSingle

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
# data_dir = './tomo_data/datasets_encoder_TF_24-03-23'
data_dir = './tomo_data/datasets_encoder_TF_08-11-23'

timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

DATA_LOAD_METHOD='FAST_TENSOR' # it can be TENSOR or DATASET or FAST_TENSOR
IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8
input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']
num_Turns_Case = 1

N_TRIALS=100
TIMEOUT=60*60*1.8 # 2 hours

# var_name = 'VrfSPS'


# Train specific
train_cfg = {
    'epochs': 30,
    'strides': [2, 2],
    'cropping': [0, 0],
    'activation': 'relu',
    'pooling_size': [2, 2],
    'pooling_strides': [1, 1],
    'pooling_padding': 'valid',
    'pooling': None,
    'dropout': 0.0,
    'loss': 'mse',
    'lr': 1e-3,
    'dataset%': 1,
    'batchnorm': False,
    'conv_padding': 'same',
    'conv_batchnorm': False,
    'normalization': 'minmax',
    'img_normalize': 'off',
    'batch_size': 32,
    'use_bias': False,
    'param_space': {
        'mu': {
            'cropping': ['0,0', '6,6', '12,12'],
            'filters': ['8,16,32', '4,16,64', '32,16,8'],
            'kernel_size': ['7,7,7', '7,5,3', '7,5,5'],
            'dense_layers': ['1024,256,128', '1024,256,32', '1024,256,64'],
            'batch_size': [32],
        },
        'VrfSPS': {
            'cropping': ['0,0', '6,6', '12,12'],
            'filters': ['8,16,32', '32,16,8'],
            'kernel_size': [13, 9, 5, 3],
            'dense_layers': ['1024,512,256', '1024,256,128', '1024,256,64'],
            'batch_size': [32],
        },

        # 'phEr': {
        #     'cropping': ['0,0'],
        #     'filters': ['8,16,32', '4,8,16'],
        #     'kernel_size': [7, 5, 3],
        #     'dense_layers': ['1024,512,32', '1024,256,32'],
        #     'batch_size': [32, 128],
        # },
        # 'enEr': {
        #     'cropping': ['0,0', '6,6'],
        #     'filters': ['8,16,32', '4,8,16'],
        #     'kernel_size': [9, 7, 5],
        #     'dense_layers': ['1024,512,64', '1024,256,64'],
        #     'batch_size': [32, 128],
        # },
        # 'bl': {
        #     'cropping': ['0,0', '6,6'],
        #     'filters': ['8,16,32', '16,32,64'],
        #     'kernel_size': [7, 5, 3],
        #     'dense_layers': ['1024,512,64', '1024,256,64'],
        #     'batch_size': [32, 128],
        # },
        # 'Vrf': {
        #     'cropping': ['0,0', '6,6', '12,12'],
        #     'filters': ['8,16,32'],
        #     'kernel_size': [13, 7, 3],
        #     'dense_layers': ['1024,512,64', '1024,256,64'],
        #     'batch_size': [32, 128],
        # },
    }
}

# phEr : 1.5402208646264626e-06 and parameters: {'crp': '0,0', 'flt': '8,16,32', 'kr_sz': 7, 'lrs': '1024,512,32', 'bs': 32}. Best is trial 184 with value: 1.5402208646264626e-06.
# bl: 0.0001191571427625604 and parameters: {'crp': '0,0', 'flt': '16,32,64', 'kr_sz': 5, 'lrs': '1024,512,64', 'bs': 32}. Best is trial 40 with value: 0.0001191571427625604.

category_keys = {
    'use_bias': 'bias',
    'conv_padding': 'cnv_pad',
    'cropping': 'crp',
    'filters': 'flt',
    'kernel_size': 'kr_sz',
    'dense_layers': 'lrs',
    'batch_size': 'bs',
}

split_keys = ['cropping', 'dense_layers', 'filters', 'kernel_size']
    

def train_test_model(var_name, x_train, y_train, x_valid, y_valid, param_space, trial):
    hparams = {}
    for var in param_space:
        value = trial.suggest_categorical(category_keys[var], param_space[var])
        if var in split_keys and isinstance(value, str):
            value = [int(i.strip()) for i in value.split(',') if i != '']
        hparams[var] = value

    # hparams = {
    #     'use_bias': trial.suggest_categorical(category_keys['use_bias'], param_space['use_bias']),
    #     'conv_padding': trial.suggest_categorical('cnv_pad', param_space['conv_padding']),
    #     'cropping': [int(i) for i in trial.suggest_categorical('crp', param_space['cropping']).split(',') if i != ''],
    #     'filters': [int(i) for i in trial.suggest_categorical('flt', param_space['filters']).split(',') if i != ''],
    #     # 'kernel_size': [int(i) for i in trial.suggest_categorical('kr_sz', param_space['kernel_size']).split(',') if i != ''],
    #     'kernel_size': trial.suggest_categorical('kr_sz', param_space['kernel_size']),
    #     'dense_layers': [int(i) for i in trial.suggest_categorical('lrs', param_space['dense_layers']).split(',') if i != ''],
    # }

    cfg = train_cfg.copy()
    cfg.update(hparams)

    model = EncoderSingle(input_shape=input_shape,
                          output_name=var_name,
                          **cfg)

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

    param_space = train_cfg.get('param_space', {})
    input_config_file = args.config
    if input_config_file:
        with open(input_config_file) as f:
            input_config = yaml.load(f, Loader=yaml.FullLoader)
        # print(input_config)
        train_cfg = input_config['encoder']
        param_space = train_cfg.get('param_space', {})
        timestamp = input_config['timestamp']
    
    print('Configuration:')
    for k, v in train_cfg.items():
        print(k, v)

    print('Model specific configuration:')
    for var in param_space.keys():
        print(var, param_space[var])

    # Initialize directories
    # trial_dir = './'
    trial_dir = os.path.join('./hparam_trials/', timestamp)
    # weights_dir = os.path.join(trial_dir, 'weights')
    # plots_dir = os.path.join(trial_dir, 'plots')
    # logs_dir = os.path.join(trial_dir, 'logs')
    # hparams_dir = os.path.join(trial_dir, 'hparams')

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
    if DATA_LOAD_METHOD == 'TENSOR':
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
    elif DATA_LOAD_METHOD == 'FAST_TENSOR':
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

    else:
        exit('DATA_LOAD_METHOD not recognised')

    print(
        f'\n---- Input files have been read, elapsed: {time.time() - start_t} ----\n')


    # Model instantiation
    start_t = time.time()
    input_shape = x_train.shape[1:]
    models = {}

    for var_name in param_space.keys():
        print(f'\n---- HyperParam tuning for: {var_name} ----\n')
        
        # Load data
        train = tf.gather(y_train, var_names.index(var_name), axis=1)
        valid = tf.gather(y_valid, var_names.index(var_name), axis=1)
        param_space = param_space.get(var_name, {})

        study = optuna.create_study(study_name=f'{var_name}_{timestamp}',
            direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
        try:
            study.optimize(lambda trial: train_test_model(var_name, x_train, train, x_valid, valid, param_space, trial),
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
        fname = os.path.join(hparams_dir, f'{var_name}_{timestamp}.csv')
        all_trials.to_csv(fname)

        # Save some plots
        fig = plot_optimization_history(study)
        fig.write_image(os.path.join(plots_dir, f'{var_name}_optimization_history.png'))

        fig = plot_parallel_coordinate(study)
        fig.write_image(os.path.join(plots_dir, f'{var_name}_parallel_coordinate.png'))

        fig = plot_param_importances(study)
        fig.write_image(os.path.join(plots_dir, f'{var_name}_param_importances.png'))

        fig = plot_slice(study)
        fig.write_image(os.path.join(plots_dir, f'{var_name}_slice.png'))

        fig = plot_contour(study)
        fig.write_image(os.path.join(plots_dir, f'{var_name}_contour.png'))

        # save the study object
        fname = os.path.join(trial_dir, f'{var_name}_{timestamp}_study.pkl')
        with open(fname, 'wb') as handle:
            pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
