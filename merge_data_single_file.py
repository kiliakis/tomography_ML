import os
import pickle as pk
import time
import tensorflow as tf
import numpy as np
from utils import sample_files, encoder_files_to_tensors
from utils import decoder_files_to_tensors, encdec_files_to_tensors


# data_dir = './tomo_data/datasets_encoder_TF_24-03-23'
data_dir = './tomo_data/datasets_decoder_TF_24-03-23'

percent = 1
normalization = 'minmax'
img_normalize = 'off'
ps_normalize = 'off'
file_chunk = 20000
# model_type = 'decoder' # Can be encoder or decoder
model_type = 'encdec'  # Can be encoder or decoder


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
    if model_type == 'encoder':
        x, y = encoder_files_to_tensors(
            file_names, normalization=normalization, img_normalize=img_normalize)
        # Saving
        print('Saving training data')
        np.savez_compressed(
            os.path.join(ML_dir, 'training-00.npz'), x=x.numpy(), y=y.numpy())

    elif model_type == 'decoder':
        for i in range(0, len(file_names), file_chunk):
            x, y = decoder_files_to_tensors(
                file_names[i: i+file_chunk], normalization=normalization, ps_normalize=ps_normalize)
            # Saving
            print(f'Saving training data: {i}-{i+file_chunk}')
            np.savez_compressed(os.path.join(
                ML_dir, f'training-{int(i//file_chunk):02d}.npz'), x=x.numpy(), y=y.numpy())
    elif model_type == 'encdec':
        for i in range(0, len(file_names), file_chunk):
            wf, turns, latents, pss = encdec_files_to_tensors(
                file_names[i: i+file_chunk], normalization=normalization,
                img_normalize=img_normalize, ps_normalize=ps_normalize)
            # Saving
            print(f'Saving training data: {i}-{i+file_chunk}')
            np.savez_compressed(os.path.join(
                ML_dir, f'encdec-training-{int(i//file_chunk):02d}.npz'), 
                WFs=wf.numpy(), turns=turns.numpy(), latents=latents.numpy(),
                PSs=pss.numpy())

    print('Done saving')

    print('Loading Validation files')
    file_names = sample_files(VALIDATION_PATH, percent)
    print('Number of Validation files: ', len(file_names))
    if model_type == 'encoder':
        x, y = encoder_files_to_tensors(
            file_names, normalization=normalization, img_normalize=img_normalize)
        np.savez_compressed(
            os.path.join(ML_dir, 'validation-00.npz'), x=x.numpy(), y=y.numpy())

    elif model_type == 'decoder':
        for i in range(0, len(file_names), file_chunk):
            x, y = decoder_files_to_tensors(
                file_names[i: i+file_chunk], normalization=normalization, ps_normalize=ps_normalize)
            # Saving
            print(f'Saving validation data: {i}-{i+file_chunk}')
            np.savez_compressed(os.path.join(
                ML_dir, f'validation-{int(i//file_chunk):02d}.npz'), x=x.numpy(), y=y.numpy())
    elif model_type == 'encdec':
        for i in range(0, len(file_names), file_chunk):
            wf, turns, latents, pss = encdec_files_to_tensors(
                file_names[i: i+file_chunk], normalization=normalization,
                img_normalize=img_normalize, ps_normalize=ps_normalize)
            # Saving
            print(f'Saving validation data: {i}-{i+file_chunk}')
            np.savez_compressed(os.path.join(
                ML_dir, f'encdec-validation-{int(i//file_chunk):02d}.npz'), 
                WFs=wf.numpy(), turns=turns.numpy(), latents=latents.numpy(),
                PSs=pss.numpy())
    print('Done saving')

    print('Loading Testing files')
    file_names = sample_files(TESTING_PATH, percent)
    print('Number of Testing files: ', len(file_names))
    if model_type == 'encoder':
        x, y = encoder_files_to_tensors(
            file_names, normalization=normalization, img_normalize=img_normalize)
        np.savez_compressed(
            os.path.join(ML_dir, 'testing-00.npz'), x=x.numpy(), y=y.numpy())
    elif model_type == 'decoder':
        for i in range(0, len(file_names), file_chunk):
            x, y = decoder_files_to_tensors(
                file_names[i: i+file_chunk], normalization=normalization, ps_normalize=ps_normalize)
            # Saving
            print(f'Saving testing data: {i}-{i+file_chunk}')
            np.savez_compressed(os.path.join(
                ML_dir, f'testing-{int(i//file_chunk):02d}.npz'), x=x.numpy(), y=y.numpy())
    elif model_type == 'encdec':
        for i in range(0, len(file_names), file_chunk):
            wf, turns, latents, pss = encdec_files_to_tensors(
                file_names[i: i+file_chunk], normalization=normalization,
                img_normalize=img_normalize, ps_normalize=ps_normalize)
            # Saving
            print(f'Saving testing data: {i}-{i+file_chunk}')
            np.savez_compressed(os.path.join(
                ML_dir, f'encdec-testing-{int(i//file_chunk):02d}.npz'), 
                WFs=wf.numpy(), turns=turns.numpy(), latents=latents.numpy(),
                PSs=pss.numpy())
    
    print('Done saving')
