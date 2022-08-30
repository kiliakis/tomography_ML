# Evaluate the ML model

import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import time
from utils import load_model_data_new, unnormalize_params, assess_decoder
from models import extendedCED, mse_loss_encoder, mse_loss_decoder

# Initialize parameters
save_dir = '/eos/user/k/kiliakis/tomo_data/datasets'
weights_dir = '/eos/user/k/kiliakis/tomo_data/weights'

# Data specific
IMG_OUTPUT_SIZE = 128
BUFFER_SIZE = 50
BATCH_SIZE = 32  # 8
latent_dim = 7 # 6 + the new VrfSPS
additional_latent_dim = 1

cnn_filters = [32, 64, 128, 256, 512, 1024]

if __name__ == '__main__':
    
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
    ML_dir = os.path.join(save_dir, 'ML_data')
    # TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    # VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')
    TESTING_PATH = os.path.join(ML_dir, 'TESTING')
    # assert os.path.exists(TRAINING_PATH)
    # assert os.path.exists(VALIDATION_PATH)
    assert os.path.exists(TESTING_PATH)

    # Create the datasets

    test_dataset = tf.data.Dataset.list_files(TESTING_PATH + '/*.pk')
    test_dataset = test_dataset.map(lambda x: tf.py_function(load_model_data_new, [x],
                                                             [tf.float32, tf.float32, tf.float32, tf.string,
                                                             tf.float32, tf.float32, tf.float32, tf.float32,
                                                             tf.float32, tf.float32, tf.float32, tf.float32,
                                                             tf.float32]))

    test_dataset = test_dataset.shuffle(BUFFER_SIZE)
    test_dataset = test_dataset.batch(BATCH_SIZE)

    # Model instantiation

    input_shape = (IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE, 1)

    eCED = extendedCED(latent_dim, additional_latent_dim, input_shape,
                       filters=cnn_filters)

    print(eCED.encoder.summary())
    print(eCED.extender.summary())
    print(eCED.decoder.summary())

    # Load weights
    eCED.encoder.load_weights(os.path.join(
        weights_dir, 'eCED_weights_encoder.h5'))
    eCED.extender.load_weights(os.path.join(
        weights_dir, 'eCED_weights_extender.h5'))
    eCED.decoder.load_weights(os.path.join(
        weights_dir, 'eCED_weights_decoder.h5'))

    # Evaluate the model
    expected_params, predicted_params = [], []
    for n, (normTurns, Timgs, PSimgs, fns, phErs, enErs, bls, intens, Vrfs, mus,
            T_normFactors, B_normFactors) in test_dataset.enumerate():
        for i in range(len(fns)):
            normTurn = normTurns[i:i+1]
            Timg = Timgs[i:i+1]
            PSimg = PSimgs[i]
            fn = fns[i]
            phEr = phErs[i].numpy()
            enEr = enErs[i].numpy()
            bl = bls[i].numpy()
            inten = intens[i].numpy()
            Vrf = Vrfs[i].numpy()
            mu = mus[i].numpy()
            T_normFactor = T_normFactors[i].numpy()
            B_normFactor = B_normFactors[i].numpy()
            preds, parss = eCED.predictPS(Timg, normTurn)
            pred_phEr, pred_enEr, pred_bl, pred_inten, pred_Vrf, pred_mu = \
                unnormalize_params(*list(parss[0].numpy()))
            expected_params.append([phEr, enEr, bl, inten, Vrf, mu])
            predicted_params.append([pred_phEr, pred_enEr, pred_bl, pred_inten,
                                     pred_Vrf, pred_mu])
    expected_params = np.array(expected_params)
    predicted_params = np.array(predicted_params)

    delParam = predicted_params-expected_params
    delParam.shape
    plt.figure()
    plt.hist(delParam[:, 0], range=(-2, 2))
    plt.xlabel('Phase Error Diff [deg]')
    plt.figure()
    plt.hist(delParam[:, 1], range=(-5, 5))
    plt.xlabel('Energy Error Diff [MeV]')
    plt.figure()
    plt.hist(delParam[:, 2]*1e12, range=(-40, 40))
    plt.xlabel('length Diff [ps]')
    plt.figure()
    plt.hist(delParam[:, 3]*1e-9, range=(-1.5, 1.5))
    plt.xlabel('I diff [1e9 prot]')
    plt.figure()
    plt.hist(delParam[:, 4], range=(-0.3, 0.3))
    plt.xlabel('Vrf diff [MV]')
    plt.figure()
    plt.hist(delParam[:, 5], range=(-0.5, 0.5))
    plt.xlabel('mu diff [a.u.]')
