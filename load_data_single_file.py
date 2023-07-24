import os
import time
from utils import fast_tensor_load

data_dir = './tomo_data/datasets_decoder_TF_24-03-23'
percent = 0.5

if __name__ == '__main__':
    # Initialize train/ test / validation paths
    ML_dir = os.path.join(data_dir, 'ML_data')
    TRAINING_PATH = os.path.join(ML_dir, 'training-??.npz')
    VALIDATION_PATH = os.path.join(ML_dir, 'validation-??.npz')
    TESTING_PATH = os.path.join(ML_dir, 'testing-??.npz')

    print('Loading Training Data')
    start_t = time.time()
    x_train, y_train = fast_tensor_load(TRAINING_PATH, percent)
    end_t = time.time()
    print(f'Num points: {len(y_train)}, Loading time: {end_t - start_t}')

    print('Loading Validation Data')
    start_t = time.time()
    x_valid, y_valid = fast_tensor_load(VALIDATION_PATH, percent)
    end_t = time.time()
    print(f'Num points: {len(y_valid)}, Loading time: {end_t - start_t}')

    print('Loading Testing Data')
    start_t = time.time()
    x_test, y_test = fast_tensor_load(TESTING_PATH, percent)
    end_t = time.time()
    print(f'Num points: {len(y_test)}, Loading time: {end_t - start_t}')

