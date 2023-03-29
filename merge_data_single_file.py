import os
import pickle as pk
import time
import tensorflow as tf
import numpy as np
from utils import sample_files, encoder_files_to_tensors


data_dir = './tomo_data/datasets_encoder_TF_24-03-23'
percent = 1
normalization = 'minmax'
img_normalize = 'off'

if __name__ == '__main__':
    # Initialize train/ test / validation paths
    ML_dir = os.path.join(data_dir, 'ML_data')
    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')
    TESTING_PATH = os.path.join(ML_dir, 'TESTING')

    assert os.path.exists(TRAINING_PATH)
    assert os.path.exists(VALIDATION_PATH)
    assert os.path.exists(TESTING_PATH)

    print('Loading Training files')
    file_names = sample_files(TRAINING_PATH, percent)
    print('Number of Training files: ', len(file_names))
    x, y = encoder_files_to_tensors(
        file_names, normalization=normalization, img_normalize=img_normalize)
    # Saving
    print('Saving training data')
    np.savez_compressed(os.path.join(ML_dir, 'training.npz'), x=x.numpy(), y=y.numpy())

    print('Loading Validation files')
    file_names = sample_files(VALIDATION_PATH, percent)
    print('Number of Validation files: ', len(file_names))
    x, y = encoder_files_to_tensors(
        file_names, normalization=normalization, img_normalize=img_normalize)
    # Saving
    print('Saving validation data')
    np.savez_compressed(
        os.path.join(ML_dir, 'validation.npz'), x=x.numpy(), y=y.numpy())
    
    print('Loading Testing files')
    file_names = sample_files(TESTING_PATH, percent)
    print('Number of Testing files: ', len(file_names))
    x, y = encoder_files_to_tensors(
        file_names, normalization=normalization, img_normalize=img_normalize)
    # Saving
    print('Saving testing data')
    np.savez_compressed(
        os.path.join(ML_dir, 'testing.npz'), x=x.numpy(), y=y.numpy())
