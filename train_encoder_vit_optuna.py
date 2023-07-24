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
from models import EncoderSingleViT, EncoderSingle

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
data_dir = './tomo_data/datasets_encoder_TF_24-03-23'
timestamp = datetime.now().strftime("%Y_%m_%d_%H-%M-%S")

MODEL_TYPE = 'Single'  # Can be SingleVit or Single
DATA_LOAD_METHOD = 'FAST_TENSOR'  # it can be TENSOR or DATASET or FAST_TENSOR
IMG_OUTPUT_SIZE = 128
BATCH_SIZE = 32  # 8
input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

var_names = ['phEr', 'enEr', 'bl',
             'inten', 'Vrf', 'mu', 'VrfSPS']
num_Turns_Case = 1

N_TRIALS = 100
TIMEOUT = 60*60*1  # 12 hours

train_cfg = {
    'epochs': 50,
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
    'normalization': 'minmax',
    'img_normalize': 'off',
    'batch_size': 32,
    'use_bias': False,
    'conv_padding': 'same',
    'param_space': {
        'VrfSPS':
        {
            'filters': ['8,16', '16,32'],
            'kernel_size': ['9,7', '7,5', '5,3'],
            'dense_layers': ['1024,512,128', '1024,512,64', '1024,256,64'],
            'batch_size': [16, 32, 64]
        },
        # {
        #     'filters': ['4,8,16', '8,16,32'],
        #     'kernel_size': ['9,7,7', '7,5,5', '5,3,3'],
        #     'dense_layers': ['1024,512,128', '1024,512,64', '1024,256,64'],
        #     'batch_size': [16, 32, 64]
        # },
    }
}
split_keys = ['cropping', 'dense_layers', 'filters', 'kernel_size']


# Train specific
# train_cfg = {
#     'epochs': 10,
#     'cropping': 12,
#     'patch_size': 8,
#     'projection_dim': 16,
#     'transformer_layers': 4,
#     'num_heads': 20,
#     'activation': 'relu',
#     'transformer_units': [1024, 256, 16],
#     'mlp_head_units': [128, 32],
#     'dropout_attention': 0.,
#     'dropout_mlp': 0.0,
#     'dropout_representation': 0.0,
#     'dropout_final': 0.0,
#     'learning_rate': 1e-3,
#     'batch_size': 32,
#     'dataset%': 1,
#     'loss': 'mse',
#     'use_bias': False,
#     'normalization': 'minmax',
#     'img_normalize': 'off',
#     'ps_normalize': 'off',
#     'batch_size': 32,
#     'final_activation': 'linear',
#     'param_space': {
#         # },
#         # 'mu': {
#         #     'transformer_units': ['16', '64,16', '128,16', '256,16'],
#         #     'transformer_layers': [1, 2, 4, 6],
#         #     'mlp_head_units': ['16', '64', '128', '256'],
#         # },
#         'VrfSPS': {
#             # 'transformer_units': ['1024,256,16'],
#             # 'transformer_layers': [4],
#             # 'mlp_head_units': ['128,32'],
#             # 'num_heads': [16, 18, 20],
#             # 'cropping': [4, 12],
#             'patch_size': ['1,26', '2,13', '2,26', '4,13', '1,52', '2,52'],
#         },
#     }
# }
# split_keys = ['transformer_units', 'mlp_head_units', 'patch_size']


category_keys = {
    'transformer_units': 'tf_unts',
    'transformer_layers': 'tf_lrs',
    'patch_size': 'ptch_sz',
    'projection_dim': 'prj_dim',
    'mlp_head_units': 'mlp_unts',
    'num_heads': 'num_hds',
    'activation': 'actv',
    'final_activation': 'fnl_actv',
    'cropping': 'crp',
    'use_bias': 'bias',
    'conv_padding': 'cnv_pad',
    'filters': 'flt',
    'kernel_size': 'kr_sz',
    'dense_layers': 'lrs',
    'batch_size': 'btch_sz'
}


def train_test_model(var_name, x_train, y_train, x_valid, y_valid, param_space, trial):
    hparams = {}
    for var in param_space:
        value = trial.suggest_categorical(
            category_keys.get(var, var), param_space[var])
        if var in split_keys and isinstance(value, str):
            value = [int(i.strip()) for i in value.split(',') if i != '']
        hparams[var] = value

    cfg = train_cfg.copy()
    cfg.update(hparams)

    if MODEL_TYPE == 'SingleVit':
        model = EncoderSingleViT(input_shape=input_shape,
                                 output_name=var_name,
                                 **cfg)
    else:
        model = EncoderSingle(input_shape=input_shape,
                              output_name=var_name,
                              **cfg)

    model.model.fit(
        x=x_train, y=y_train,
        epochs=cfg['epochs'],
        validation_data=(x_valid, y_valid),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=5),
            TFKerasPruningCallback(trial, "val_loss")],
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
    tf.random.set_seed(0)

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
                                    direction='minimize', pruner=optuna.pruners.MedianPruner())
        try:
            study.optimize(lambda trial: train_test_model(var_name, x_train, train, x_valid, valid, param_space, trial),
                           gc_after_trial=True, n_jobs=1, n_trials=N_TRIALS, timeout=TIMEOUT)
        except KeyboardInterrupt:
            print('KeyboardInterrupt')

        pruned_trials = study.get_trials(
            deepcopy=False, states=[TrialState.PRUNED])
        complete_trials = study.get_trials(
            deepcopy=False, states=[TrialState.COMPLETE])

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
        fig.write_image(os.path.join(
            plots_dir, f'{var_name}_optimization_history.png'))

        fig = plot_parallel_coordinate(study)
        fig.write_image(os.path.join(
            plots_dir, f'{var_name}_parallel_coordinate.png'))

        fig = plot_param_importances(study)
        fig.write_image(os.path.join(
            plots_dir, f'{var_name}_param_importances.png'))

        fig = plot_slice(study)
        fig.write_image(os.path.join(plots_dir, f'{var_name}_slice.png'))

        fig = plot_contour(study)
        fig.write_image(os.path.join(plots_dir, f'{var_name}_contour.png'))

        # save the study object
        fname = os.path.join(trial_dir, f'{var_name}_{timestamp}_study.pkl')
        with open(fname, 'wb') as handle:
            pickle.dump(study, handle, protocol=pickle.HIGHEST_PROTOCOL)
