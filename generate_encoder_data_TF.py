# GENERATE ML DATA
from genericpath import exists
import os
import numpy as np
import pickle as pk
import random
from utils import extract_data_Fromfolder
from utils import loadTF, calc_bin_centers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

IMG_OUTPUT_SIZE = 128
zeropad = 14
start_turn = 1  # skip first turn from theo simulations
skipturns = 3

# Read normalized sim data or generate them?
# readSimData = True
# saveAllData = False
# eos = '/home/kiliakis/cernbox'
eos = '/eos/user/k/kiliakis/'

simulations_dir = eos + '/tomo_data/results_tomo_02-12-22'
save_dir = eos + '/tomo_data/datasets_encoder_TF_16-12-22'

# For traning, test and validation, out of all cases simulated (9229)
num_Cases = -1

skip_first = 12024  # skip the first simulation dirs, useful for resuming after a crash
last = 12448
# out of the 100 turns selected by case (1 out of 3, so in max 300 turns)
# num_Turns_Case = 50
# num_Turns_Case_test = 1
training_ratio = 0.90


if __name__ == '__main__':

    # This part is related to the TF convolution
    cut_left = 0
    cut_right = 2*np.pi
    n_slices = 100

    timescale = calc_bin_centers(cut_left, cut_right, n_slices)
    tf_path = eos + '/tomo_data/transfer_functions/TF_B{}.h5'
    freq_array, TF_array = loadTF(path=tf_path)
    
    # fig = plt.figure()
    # plt.plot(freq_array, TF_array.real, label='real')
    # plt.plot(freq_array, TF_array.imag, label='imag')    
    # # plt.plot(freq_array, np.abs(TF_array), label='abs')
    # plt.legend()
    # plt.savefig('plots/transfer_function.jpg', dpi=400)
    # plt.close()
    
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

    train_dirs, valid_dirs = train_test_split(train_dirs,
                                              train_size=training_ratio,
                                              random_state=1)
    i = 0
    for SAVE_PATH, data_dirs in [(TRAINING_PATH, train_dirs),
                                 (VALIDATION_PATH, valid_dirs),
                                 (TESTING_PATH, test_dirs)]:
        print('Saving data to: ', SAVE_PATH)
        for fn in data_dirs:
            if i < skip_first:
                # i += num_Turns_Case
                i += 1
                continue
            if last > 0 and i >=last:
                break
            print(i, fn)
            try:
                paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
                    extract_data_Fromfolder(fn, simulations_dir, IMG_OUTPUT_SIZE, zeropad,
                                            start_turn, skipturns, version=4,
                                            time_scale=timescale, freq_array=freq_array,
                                            TF_array=TF_array)
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
                           #    'B_img': PS_img_dec
                           }

            # for turn in [0]+random.choices(normSimDict['turns'][1:], k=num_Turns_Case):
            tenK = os.path.join(SAVE_PATH, f'{int(i // 10000):02d}x10K')
            os.makedirs(tenK, exist_ok=True)
            pk.dump({
                    'turn': 0,
                    'T_img': normSimDict['T_img'],
                    'params': normSimDict['params'],
                    'fn': normSimDict['fn'],
                    'PS': 0},
                    open(os.path.join(tenK, "{:06d}.pk".format(i)), "wb"))

            i += 1
            
