import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import glob
from mlp_lhc_tomography.utils import read_pk
from mlp_lhc_tomography.utils import normalizeIMG, minMaxScaleIMG
from mlp_lhc_tomography.utils import normalize_params, minmax_normalize_param

def get_best_model_timestamp(path, model='enc'):
    from sort_trial_summaries import extract_trials
    header, rows = extract_trials(path)
    for row in rows:
        if model in row[header.index('model')]:
            return row[header.index('date')]


def plot_loss(lines, title='', figname=None):
    plt.figure()
    plt.title(title)
    for line in lines.keys():
        if 'val' in line.lower():
            marker = 'x'
        else:
            marker = '.'
        plt.semilogy(lines[line], marker=marker, label=line)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(ncol=2)
    plt.tight_layout()
    if figname:
        plt.savefig(figname, dpi=300)
        plt.close()

def tomoscope_files_to_tensors(files, normalize=True, normalization='default',
                            img_normalize='default', ps_normalize='default',
                            num_turns=1):
    waterfall_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    turn_arr = np.zeros((len(files), num_turns), dtype=np.float32)
    latent_arr = np.zeros((len(files), 7), dtype=np.float32)
    phasespace_arr = np.zeros((len(files), 128, 128, num_turns), dtype=np.float32)
    i = 0
    keep_idx = []
    for file in files:
        try:
            waterfall, turn, latents, ps = load_tomoscope_data(file,
                                                            normalize=normalize,
                                                            normalization=normalization,
                                                            img_normalize=img_normalize,
                                                            ps_normalize=ps_normalize)
        except Exception as e:
            print(f'Skipping file {file}, ', e)
            continue
        if not keep_idx:
            assert len(turn) >= num_turns
            keep_idx = np.arange(0, len(turn), len(turn) // num_turns)
        assert len(keep_idx) == num_turns

        waterfall_arr[i] = waterfall
        turn_arr[i] = turn[keep_idx]
        latent_arr[i] = latents
        phasespace_arr[i] = ps[:, :, keep_idx]
        i += 1
    waterfall_arr = tf.convert_to_tensor(waterfall_arr[:i])
    turn_arr = tf.convert_to_tensor(turn_arr[:i])
    latent_arr = tf.convert_to_tensor(latent_arr[:i])
    phasespace_arr = tf.convert_to_tensor(phasespace_arr[:i])

    return waterfall_arr, turn_arr, latent_arr, phasespace_arr


def load_tomoscope_data(pk_file, normalization, normalize=True, img_normalize='default',
                     ps_normalize='default'):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    # PS = np.reshape(PS, PS.shape+(1,))
    if ps_normalize == 'default':
        PS = PS / params_dict['B_normFactor']
    elif ps_normalize == 'off':
        pass
    # turn_num = normalizeTurn(turn_num)
    turn_num = minmax_normalize_param(turn_num, 1, 298, target_range=(0, 1))
    if img_normalize == 'default':
        T_img = normalizeIMG(T_img)
    elif img_normalize == 'minmax':
        T_img = minMaxScaleIMG(T_img)
    elif img_normalize == 'off':
        pass
    phEr = float(params_dict['phEr'])
    enEr = float(params_dict['enEr'])
    bl = float(params_dict['bl'])
    inten = float(params_dict['int'])
    Vrf = float(params_dict['Vrf'])
    mu = float(params_dict['mu'])
    VrfSPS = float(params_dict['VrfSPS'])
    if normalize:
        phEr, enEr, bl, inten, Vrf, mu, VrfSPS = normalize_params(
            phEr, enEr, bl, inten, Vrf, mu, VrfSPS, normalization=normalization)
    # T_normFactor = float(params_dict['T_normFactor'])
    # B_normFactor = float(params_dict['B_normFactor'])
    # return turn_num, T_img, PS, fn, phEr, enEr, bl, inten, Vrf, mu, VrfSPS, T_normFactor, B_normFactor
    return (T_img, turn_num, [phEr, enEr, bl, inten, Vrf, mu, VrfSPS], PS)


def fast_tensor_load_encdec(path, percent=1.0, max_files=-1):
    wf_arr, turn_arr, latent_arr, ps_arr = [], [], [], []
    all_files = glob.glob(path)
    if max_files > 0:
        all_files = all_files[max_files]
    # For every file that matches the regexp
    print(len(all_files))
    for file in all_files:
        print(f'Loading {file}')
        # decompress and load file
        with np.load(file) as data:
            wf, turn, latent, ps = data['WFs'], data['turns'], data['latents'], data['PSs']
        print('ps shape:', ps.shape)
        # Keep a smaller percentage if needed
        if percent < 1 and percent > 0:
            points = len(wf)
            keep_points = np.random.choice(
                points, int(points * percent), replace=False)
            wf, turn, latent, ps = wf[keep_points], turn[keep_points], latent[keep_points], ps[keep_points]
        # append to list
        wf_arr.append(tf.convert_to_tensor(wf))
        turn_arr.append(tf.convert_to_tensor(turn))
        latent_arr.append(tf.convert_to_tensor(latent))
        ps_arr.append(tf.convert_to_tensor(ps))

    # make the final tensor
    wf_arr = tf.concat(wf_arr, axis=0)
    turn_arr = tf.concat(turn_arr, axis=0)
    latent_arr = tf.concat(latent_arr, axis=0)
    ps_arr = tf.concat(ps_arr, axis=0)

    return wf_arr, turn_arr, latent_arr, ps_arr