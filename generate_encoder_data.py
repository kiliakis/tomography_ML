# GENERATE ML DATA
import os
import numpy as np
import pickle as pk
from utils import extract_data_Fromfolder
from utils import loadTF, calc_bin_centers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse

IMG_OUTPUT_SIZE = 128
zeropad = 14
start_turn = 1  # skip first turn from theo simulations
skipturns = 3

# Input output directories
eos = '/eos/user/k/kiliakis/'
simulations_dir = eos + '/tomo_data/results_tomo_02-12-22'
save_dir = eos + '/tomo_data/datasets_encoder_TF_03-02-23'

parser = argparse.ArgumentParser(
    description='Generate the encoder data from the raw simulation data')

parser.add_argument('-f', '--first', type=int, default=0,
                    help='The first simulation dir.')

parser.add_argument('-l', '--last', type=int, default=-1,
                    help='The last simulation dir to use. -1 to process all of them.')

parser.add_argument('-d', '--dry-run', type=int, default=0,
                    help='Only collect stats, do not actually create the directories.')

parser.add_argument('-tf', '--transfer-function', type=int, default=1,
                    help='Apply transfer function.')

parser.add_argument('-train', '--train-ratio', type=float, default=0.85,
                    help='The ratio of training samples/ all samples.')


# E_normFactor = 23231043000.0
E_normFactor = 25000000000.0
# B_normFactor = 768246000.0
B_normFactor = 800000000.0
# T_normFactor = 25860460000.0
T_normFactor = 28000000000.0

if __name__ == '__main__':
    args = parser.parse_args()
    skip_first = args.first
    skip_last  = args.last
    dry_run = args.dry_run
    apply_tf = args.transfer_function
    training_ratio = args.train_ratio

    if apply_tf:
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

    # Split dirs in train, test and validation sets
    train_dirs, test_dirs = train_test_split(all_sim_dirs,
                                             train_size=training_ratio,
                                             random_state=1)

    test_dirs, valid_dirs = train_test_split(test_dirs,
                                             train_size=0.5,
                                             random_state=1)
    i = 0
    E_maxs = []
    T_maxs = []
    B_maxs = []

    for SAVE_PATH, data_dirs in [(TRAINING_PATH, train_dirs),
                                 (VALIDATION_PATH, valid_dirs),
                                 (TESTING_PATH, test_dirs)]:
        print('Saving data to: ', SAVE_PATH)
        for fn in data_dirs:
            if i < skip_first:
                i += 1
                continue
            if skip_last > 0 and i >= skip_last:
                break
            print(i, fn)
            try:
                if apply_tf:
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
                i+=1
                continue

            # Get max values
            # E_normFactor = np.max(E_img)
            # T_normFactor = np.max(T_img)
            # B_normFactor = np.max(PS_img_dec.flatten())

            E_maxs.append(E_normFactor)
            T_maxs.append(T_normFactor)
            B_maxs.append(B_normFactor)

            if not dry_run:
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
                            'turns': sel_turns,
                            'T_img': T_img,
                            #    'skipturns': skipturns,
                            #    'E_img': E_img,
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

    print(f'E_max: {np.max(E_normFactor)}')
    print(f'T_max: {np.max(T_normFactor)}')
    print(f'B_max: {np.max(B_normFactor)}')