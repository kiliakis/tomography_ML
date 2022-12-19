# GENERATE ML DATA
import os
import numpy as np
import pickle as pk
import random
from utils import extract_data_Fromfolder
from sklearn.model_selection import train_test_split
from utils import loadTF, calc_bin_centers
import argparse

parser = argparse.ArgumentParser(description='Generate decoder training/validation/testing data',
                                 usage='python generate_decoder_data.py')

parser.add_argument('-s', '--skip-first', type=int, default=0,
                    help='Skip the first cases.'
                    ' Default: 0')

parser.add_argument('-l', '--last', type=int, default=-1,
                    help='Last case to generate.'
                    ' Default: -1 (generate all data)')

IMG_OUTPUT_SIZE = 128
zeropad = 14
start_turn = 1  # skip first turn from theo simulations
skipturns = 3
APPLY_TF = True

# Read normalized sim data or generate them?
# readSimData = True
# saveAllData = False
eos = '/eos/user/k/kiliakis'
simulations_dir = eos + '/tomo_data/results_tomo_02-12-22'
# save_dir = eos + '/tomo_data/datasets_decoder_02-12-22'
save_dir = eos + '/tomo_data/datasets_decoder_TF_16-12-22'

# For traning, test and validation, out of all cases simulated (9229)
num_Cases = -1
skip_first = 0  # skip the first simulation dirs, useful for resuming after a crash
last = -1
# out of the 100 turns selected by case (1 out of 3, so in max 300 turns)
num_Turns_Case = 20
num_Turns_Case_test = 1
training_ratio = 0.90

if __name__ == '__main__':
    args = parser.parse_args()
    skip_first = args.skip_first
    last = args.last

    if APPLY_TF:
        # This part is related to the TF convolution
        cut_left = 0
        cut_right = 2*np.pi
        n_slices = 100

        timescale = calc_bin_centers(cut_left, cut_right, n_slices)
        tf_path = eos + '/tomo_data/transfer_functions/TF_B{}.h5'
        freq_array, TF_array = loadTF(path=tf_path)

    ML_dir = os.path.join(save_dir, 'ML_data')
    TRAINING_PATH = os.path.join(ML_dir, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir, 'VALIDATION')
    TESTING_PATH = os.path.join(ML_dir, 'TESTING')
    os.makedirs(TRAINING_PATH, exist_ok=True)
    os.makedirs(TESTING_PATH, exist_ok=True)
    os.makedirs(VALIDATION_PATH, exist_ok=True)

    print('Generating ML data from Theo simulations, importing it from ',
          simulations_dir)
    # Get list of all sim directories
    all_sim_dirs = os.listdir(simulations_dir)

    # Keep the first num_Cases dirs
    if num_Cases > 0 and len(all_sim_dirs) > num_Cases:
        all_sim_dirs = all_sim_dirs[:num_Cases]

    # Split dirs in train, test and validation sets
    train_dirs, test_dirs = train_test_split(all_sim_dirs,
                                             train_size=training_ratio,
                                             random_state=1)

    test_dirs, valid_dirs = train_test_split(test_dirs,
                                             train_size=0.5,
                                             random_state=1)
    print('Total train: ', len(train_dirs))
    print('Total test: ', len(test_dirs))
    print('Total valid: ', len(valid_dirs))

    i = 0
    for SAVE_PATH, data_dirs in [(TRAINING_PATH, train_dirs),
                                 (VALIDATION_PATH, valid_dirs),
                                 (TESTING_PATH, test_dirs)]:
        print('Saving data to: ', SAVE_PATH)
        for fn in data_dirs:
            if i < skip_first:
                i += num_Turns_Case+1
                continue
            if last > 0 and i >= last:
                break
            print(i, fn)
            try:
                if APPLY_TF:
                    paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
                        extract_data_Fromfolder(fn, simulations_dir, IMG_OUTPUT_SIZE, zeropad,
                                                start_turn, skipturns, version=4,
                                                time_scale=timescale, freq_array=freq_array,
                                                TF_array=TF_array)
                else:
                    paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
                        extract_data_Fromfolder(fn, simulations_dir, IMG_OUTPUT_SIZE, zeropad,
                                                start_turn, skipturns, version=4)
            except Exception as e:
                print('NOT VALID: ', fn, e)

            # Get max values
            E_normFactor = np.max(E_img)
            T_normFactor = np.max(T_img)
            B_normFactor = np.max(PS_img_dec.flatten())

            # Normalize data
            E_img = E_img / E_normFactor
            T_img = T_img / T_normFactor
            PS_img_dec = PS_img_dec / B_normFactor

            # Save norm factors
            paramsDict['E_normFactor'] = E_normFactor
            paramsDict['T_normFactor'] = T_normFactor
            paramsDict['B_normFactor'] = B_normFactor

            # Construct dictionary to pickle
            normSimDict = {'fn': fn,
                           'params': paramsDict,
                           #    'skipturns': skipturns,
                           'turns': sel_turns,
                           #    'E_img': E_img,
                           'T_img': T_img,
                           'B_img': PS_img_dec
                           }

            for turn in [0]+random.choices(normSimDict['turns'][1:], k=num_Turns_Case):
                tenK = os.path.join(SAVE_PATH, f'{int(i // 10000):02d}x10K')
                os.makedirs(tenK, exist_ok=True)
                pk.dump({'turn': turn,
                        'T_img': normSimDict['T_img'],
                         'params': normSimDict['params'],
                         'fn': normSimDict['fn'],
                         'PS': normSimDict['B_img'][:, :, turn//skipturns]
                         },
                        open(os.path.join(tenK, "{:06d}.pk".format(i)), "wb"))

                i += 1
