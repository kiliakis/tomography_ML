import numpy as np
import pickle as pk
import h5py as hp
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import glob


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


def normalizeIMG(img, maxPixel=1):
    return (img / (maxPixel/2)) - 1


def unnormalizeIMG(img, maxPixel=1):
    return (img+1)*(maxPixel/2)


# def normalizeTurn(turn_num, maxTurns=300.0):
#     return (turn_num / (maxTurns/2)) - 1


# def unnormalizeTurn(turn_num, maxTurns=300.0):
#     return np.round((turn_num+1)*(maxTurns/2), 0)


def minMaxScaleIMG(X, feature_range=(0, 1)):
    min_val = np.min(X)
    max_val = np.max(X)
    scale = (feature_range[1] - feature_range[0]) / (max_val - min_val)
    return scale * (X-min_val) + feature_range[0]


def minmax_normalize_param(val, min_val, max_val, target_range=(0, 1)):
    scale = (target_range[1] - target_range[0]) / (max_val - min_val)
    return scale * (val-min_val) + target_range[0]


def minmax_unnormalize_param(norm_val, min_val, max_val, norm_range=(0, 1)):
    scale = (max_val - min_val) / (norm_range[1] - norm_range[0])
    return scale * (norm_val - norm_range[0]) + min_val


def normalize_param(val, mu, sig):
    return (val-mu)/sig


def unnormalize_param(norm_val, mu, sig):
    return norm_val*sig+mu


def calc_bin_centers(cut_left, cut_right, n_slices):
    bin_centers = np.linspace(cut_left, cut_right, n_slices)
    # bin_centers = (edges[:-1] + edges[1:])/2
    return bin_centers


def loadTF(path, beam=2, cut=52):
    path = path.format(beam)
    h5file = hp.File(path, 'r')
    freq_array = np.array(h5file["/TransferFunction/freq"])
    TF_array = np.array(h5file["/TransferFunction/TF"])
    h5file.close()
    # TF_array = fitTF(freq_array,TF_array)
    filt = (20*np.log10(np.abs(TF_array)) < -cut) & (TF_array > 0)
    TF_array[filt] *= 10**(-cut/20.0) / np.abs(TF_array[filt])
    return freq_array, TF_array


def bunchProfile_TFconvolve(frames, timeScale, freq_array, TF_array):
    Nframes = frames.shape[1]
    frames_corr = np.zeros(frames.shape)
    noints0 = timeScale.shape[0]
    time = timeScale
    dt = time[1] - time[0]
    # Extending the time array to improve the deconvolution (usefull for bew bunch acquisitions)
    if noints0 < 40000:  # correspomds to 1MHz resolution, then improve it
        time = np.arange(time.min(), time.max() + 100e-9, dt)

    # Recalculate the number of points and the frequency array
    noints = time.shape[0]
    freq = np.fft.fftfreq(noints, d=dt)
    #  interpolate to the frequency array
    TF = np.interp(freq, np.fft.fftshift(freq_array), np.fft.fftshift(TF_array.real)) + \
        1j * np.interp(freq, np.fft.fftshift(freq_array),
                       np.fft.fftshift(TF_array.imag))

    for i in range(Nframes):
        profile = frames[:, i]
        profile = profile - profile[0:10].mean()
        profile = np.concatenate((profile, np.zeros(time.shape[0] - noints0)))

        # Convolution
        filtered_f = np.fft.fft(profile) * TF
        filtered = np.fft.ifft(filtered_f).real

        filtered -= filtered[0:10].mean()
        filtered *= np.max(profile)/np.max(filtered)
        frames_corr[:, i] = filtered[:frames.shape[0]]
        time_corr = time[:frames.shape[0]]

    return time_corr, frames_corr


def extract_data_Fromfolder(fn, simulations_dir, IMG_OUTPUT_SIZE, zeropad,
                            start_turn, skipturns, version=3,
                            time_scale=None, freq_array=None, TF_array=None):

    if version <= 2:
        pattern_string = 'phEr(?P<phEr>.+)_enEr(?P<enEr>.+)_bl(?P<bl>.+)_int(?P<int>.+)'
    elif version == 3:
        pattern_string = 'phEr(?P<phEr>.+)_enEr(?P<enEr>.+)_bl(?P<bl>.+)_int(?P<int>.+)_Vrf(?P<Vrf>.+)_mu(?P<mu>.+)'
    else:
        pattern_string = 'phEr(?P<phEr>.+)_enEr(?P<enEr>.+)_bl(?P<bl>.+)_int(?P<int>.+)_Vrf(?P<Vrf>.+)_mu(?P<mu>.+)_VrfSPS(?P<VrfSPS>.+)'

    paramsDict = {k: float(v) for k, v in re.match(
        pattern_string, fn).groupdict().items()}
    E_img = np.zeros((IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE))
    T_img = np.zeros((IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE))
    with hp.File(os.path.join(os.path.join(simulations_dir, fn), 'saved_result.hdf5'), 'r') as sf:
        BunchProfiles = np.array(sf['bunchProfiles'])

        # fig = plt.figure()
        # plt.plot(time_scale, BunchProfiles[:, 0], label='before_tf')
        if (time_scale is not None) and (freq_array is not None) and (TF_array is not None):
            _, BunchProfiles = bunchProfile_TFconvolve(BunchProfiles, time_scale,
                                                       freq_array, TF_array)

        BunchProfiles = BunchProfiles / sf['columns'][0][3]*paramsDict['int']

        # plt.plot(time_scale, BunchProfiles[:, 0], label='after_tf')
        # plt.legend()
        # plt.savefig('plots/profile_before_after_tf.jpg', dpi=400)
        # plt.close()

        EnergyProfiles = np.array(
            sf['energyProfiles'])/sf['columns'][0][3]*paramsDict['int']
        phaseSpace_density_array = np.array(sf['phaseSpace_density_array'])
        PS_imgs = np.zeros((IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE,
                           phaseSpace_density_array.shape[1]))
        for i in range(phaseSpace_density_array.shape[1]):
            turn_PS = np.transpose(np.reshape(
                phaseSpace_density_array[:, i], (99, 99)))/sf['columns'][0][3]*paramsDict['int']
            PS_imgs[:, :, i] = np.pad(
                turn_PS, ((zeropad, zeropad+1), (zeropad, zeropad+1)))
        sel_turns = np.arange(
            start_turn, skipturns*(IMG_OUTPUT_SIZE-2*zeropad), skipturns).astype(np.int32)
        PS_img_dec = PS_imgs[:, :, sel_turns]
        E_img = np.pad(EnergyProfiles[:, sel_turns], ((
            zeropad, zeropad), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
        T_img = np.pad(BunchProfiles[:, sel_turns], ((
            zeropad, zeropad), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
    return paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec


def read_pk(fname):
    data = pk.loads(tf.io.decode_raw(tf.io.read_file(fname), tf.uint8))
    return data['turn'], data['T_img'], data['PS'], data['fn'], data['params']

# Returns a random sample of percent filenames from input path

def sample_files(path, percent, keep_every=1):
    ret_files = []
    dirs = glob.glob(path + '/*x10K')
    seen_set = set()
    for dir in dirs:
        # Get entire list of files
        files = os.listdir(dir)

        # this loop is to make sure that the inputs are unique
        for f in files:
            file_id = int(f.split('.pk')[0]) // keep_every
            if file_id not in seen_set:
                seen_set.add(file_id)
                ret_files.append(os.path.join(dir, f))
        # sample = [os.path.join(dir, f) for f in sample]
        # ret_files += sample

    # randomly sample the list
    ret_files = np.random.choice(ret_files, int(percent * len(ret_files)),
                                 replace=False)
    # np.random.shuffle(ret_files)
    return ret_files



def load_encdec_data(pk_file, normalization, normalize=True, img_normalize='default',
                     ps_normalize='default'):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    PS = np.reshape(PS, PS.shape+(1,))
    if ps_normalize == 'default':
        PS = normalizeIMG(PS)
    elif ps_normalize == 'off':
        pass
    # turn_num = normalizeTurn(turn_num)
    turn_num = minmax_normalize_param(turn_num, 1, 298, target_range=(0, 1))
    if img_normalize=='default':
        T_img = normalizeIMG(T_img)
    elif img_normalize=='minmax':
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


def load_encoder_data(pk_file, normalization, normalize=True, img_normalize='default'):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    # PS = np.reshape(PS, PS.shape+(1,))
    # turn_num = normalizeTurn(turn_num)
    if img_normalize=='default':
        T_img = normalizeIMG(T_img)
    elif img_normalize=='minmax':
        T_img = minMaxScaleIMG(T_img)
    elif img_normalize=='off':
        pass

    # PS = normalizeIMG(PS)
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
    return (T_img, [phEr, enEr, bl, inten, Vrf, mu, VrfSPS])


def load_decoder_data(pk_file, normalization, ps_normalize='default'):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    # T_img = np.reshape(T_img, T_img.shape+(1,))
    PS = np.reshape(PS, PS.shape+(1,))
    # turn_num = normalizeTurn(turn_num)
    turn_num = minmax_normalize_param(turn_num, 1, 298, target_range=(0, 1))
    if ps_normalize == 'default':
        PS = normalizeIMG(PS)
    elif ps_normalize == 'off':
        pass

    phEr = float(params_dict['phEr'])
    enEr = float(params_dict['enEr'])
    bl = float(params_dict['bl'])
    inten = float(params_dict['int'])
    Vrf = float(params_dict['Vrf'])
    mu = float(params_dict['mu'])
    VrfSPS = float(params_dict['VrfSPS'])
    phEr, enEr, bl, inten, Vrf, mu, VrfSPS = normalize_params(
        phEr, enEr, bl, inten, Vrf, mu, VrfSPS, normalization=normalization)

    # T_normFactor = float(params_dict['T_normFactor'])
    # B_normFactor = float(params_dict['B_normFactor'])
    return ([turn_num, phEr, enEr, bl, inten, Vrf, mu, VrfSPS], PS)


# def load_model_data_new(pk_file):
#     turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
#     T_img = np.reshape(T_img, T_img.shape+(1,))
#     PS = np.reshape(PS, PS.shape+(1,))
#     # turn_num = normalizeTurn(turn_num)
#     turn_num = minmax_normalize_param(turn_num, 1, 298, target_range=(0, 1))

#     # T_img = normalizeIMG(T_img)
#     PS = normalizeIMG(PS)
#     phEr = float(params_dict['phEr'])
#     enEr = float(params_dict['enEr'])
#     bl = float(params_dict['bl'])
#     inten = float(params_dict['int'])
#     Vrf = float(params_dict['Vrf'])
#     mu = float(params_dict['mu'])
#     VrfSPS = float(params_dict['VrfSPS'])
#     T_normFactor = float(params_dict['T_normFactor'])
#     B_normFactor = float(params_dict['B_normFactor'])
#     return turn_num, T_img, PS, fn, phEr, enEr, bl, inten, Vrf, mu, VrfSPS, T_normFactor, B_normFactor


def encdec_files_to_tensors(files, normalize=True, normalization='default', 
                            img_normalize='default', ps_normalize='default'):
    waterfall_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    turn_arr = np.zeros(len(files), dtype=np.float32)
    latent_arr = np.zeros((len(files), 7), dtype=np.float32)
    phasespace_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    i = 0
    for file in files:
        try:
            waterfall, turn, latents, ps = load_encdec_data(file,
                                                        normalize=normalize,
                                                        normalization=normalization,
                                                        img_normalize=img_normalize,
                                                        ps_normalize=ps_normalize)
        except Exception as e:
            print(f'Skipping file {file}, ', e)
            continue
        waterfall_arr[i] = waterfall
        turn_arr[i] = turn
        latent_arr[i] = latents
        phasespace_arr[i] = ps
        i+=1
    waterfall_arr = tf.convert_to_tensor(waterfall_arr[:i])
    turn_arr = tf.convert_to_tensor(turn_arr[:i])
    latent_arr = tf.convert_to_tensor(latent_arr[:i])
    phasespace_arr = tf.convert_to_tensor(phasespace_arr[:i])

    return waterfall_arr, turn_arr, latent_arr, phasespace_arr


def encoder_files_to_tensors(files, normalize=True, normalization='default', img_normalize='default'):
    feature_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    output_arr = np.zeros((len(files), 7), dtype=np.float32)
    i = 0
    for file in files:
        try:
            features, output = load_encoder_data(file, normalize=normalize,
                                             normalization=normalization,
                                             img_normalize=img_normalize)
        except Exception as e:
            print(f'Skipping file {file}, ', e)
            continue
        feature_arr[i] = features
        output_arr[i] = output
        i+=1
    x_train = tf.convert_to_tensor(feature_arr[:i])
    y_train = tf.convert_to_tensor(output_arr[:i])
    return x_train, y_train


def decoder_files_to_tensors(files, normalization='default', ps_normalize='default'):
    feature_arr = np.zeros((len(files), 8), dtype=np.float32)
    output_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    i = 0
    for file in files:
        try:
            features, output = load_decoder_data(file, normalization=normalization,
                                                 ps_normalize=ps_normalize)
        except Exception as e:
            print(f'Skipping file {file}, ', e)
            continue
        feature_arr[i] = features
        output_arr[i] = output
        i+=1
    x_train = tf.convert_to_tensor(feature_arr[:i])
    y_train = tf.convert_to_tensor(output_arr[:i])
    return x_train, y_train


def normalize_params(*args, normalization):
    if normalization == 'std':
        phEr_a, phEr_b = 0.45, 29.14
        enEr_a, enEr_b = -0.11, 58.23
        bl_a, bl_b = 1.48e-9, 0.167e-9
        intens_a, intens_b = 1.565e11, 0.843e11
        Vrf_a, Vrf_b = 6.06, 1.79
        mu_a, mu_b = 2.89, 1.11
        VrfSPS_a, VrfSPS_b = 8.51, 2.02
        norm_func = normalize_param

    elif normalization == 'default':
        phEr_a, phEr_b = 0, 50
        enEr_a, enEr_b = 0, 100
        bl_a, bl_b = 1.4e-9, 0.2e-9
        intens_a, intens_b = 1.225e11, 0.37e11
        Vrf_a, Vrf_b = 6, 2.2
        mu_a, mu_b = 2, 1
        VrfSPS_a, VrfSPS_b = 8.5, 2.2
        norm_func = normalize_param

    elif normalization == 'minmax':
        phEr_a, phEr_b = -50., 50.
        enEr_a, enEr_b = -100., 100.
        bl_a, bl_b = 1.2e-9, 1.8e-9
        intens_a, intens_b = 1.0e10, 3.0e11
        Vrf_a, Vrf_b = 3., 9.2
        mu_a, mu_b = 1., 5.
        VrfSPS_a, VrfSPS_b = 5., 12.0
        norm_func = minmax_normalize_param

    if len(args) == 6:
        return norm_func(args[0], phEr_a, phEr_b),\
            norm_func(args[1], enEr_a, enEr_b),\
            norm_func(args[2], bl_a, bl_b),\
            norm_func(args[3], Vrf_a, Vrf_b),\
            norm_func(args[4], mu_a, mu_b),\
            norm_func(args[5], VrfSPS_a, VrfSPS_b)
    elif len(args) == 7:
        return norm_func(args[0], phEr_a, phEr_b),\
            norm_func(args[1], enEr_a, enEr_b),\
            norm_func(args[2], bl_a, bl_b),\
            norm_func(args[3], intens_a, intens_b),\
            norm_func(args[4], Vrf_a, Vrf_b),\
            norm_func(args[5], mu_a, mu_b),\
            norm_func(args[6], VrfSPS_a, VrfSPS_b)
        

def unnormalize_params(*args, normalization):
    if normalization == 'std':
        phEr_a, phEr_b = 0.45, 29.14
        enEr_a, enEr_b = -0.11, 58.23
        bl_a, bl_b = 1.48e-9, 0.167e-9
        intens_a, intens_b = 1.565e11, 0.843e11
        Vrf_a, Vrf_b = 6.06, 1.79
        mu_a, mu_b = 2.89, 1.11
        VrfSPS_a, VrfSPS_b = 8.51, 2.02
        unnorm_func = unnormalize_param

    elif normalization == 'default':
        phEr_a, phEr_b = 0, 50
        enEr_a, enEr_b = 0, 100
        bl_a, bl_b = 1.4e-9, 0.2e-9
        intens_a, intens_b = 1.225e11, 0.37e11
        Vrf_a, Vrf_b = 6, 2.2
        mu_a, mu_b = 2, 1
        VrfSPS_a, VrfSPS_b = 8.5, 2.2
        unnorm_func = unnormalize_param

    elif normalization == 'minmax':
        phEr_a, phEr_b = -50., 50.
        enEr_a, enEr_b = -100., 100.
        bl_a, bl_b = 1.2e-9, 1.8e-9
        intens_a, intens_b = 1.0e10, 3.0e11
        Vrf_a, Vrf_b = 3., 9.2
        mu_a, mu_b = 1., 5.
        VrfSPS_a, VrfSPS_b = 5., 12.0
        unnorm_func = minmax_unnormalize_param

    if len(args) == 6:
        return unnorm_func(args[0], phEr_a, phEr_b),\
            unnorm_func(args[1], enEr_a, enEr_b),\
            unnorm_func(args[2], bl_a, bl_b),\
            unnorm_func(args[3], Vrf_a, Vrf_b),\
            unnorm_func(args[4], mu_a, mu_b),\
            unnorm_func(args[5], VrfSPS_a, VrfSPS_b)
    elif len(args) == 7:
        return unnorm_func(args[0], phEr_a, phEr_b),\
            unnorm_func(args[1], enEr_a, enEr_b),\
            unnorm_func(args[2], bl_a, bl_b),\
            unnorm_func(args[3], intens_a, intens_b),\
            unnorm_func(args[4], Vrf_a, Vrf_b),\
            unnorm_func(args[5], mu_a, mu_b),\
            unnorm_func(args[6], VrfSPS_a, VrfSPS_b)

def assess_model(predictions, turn_normalized, T_image, PS_image,
                 plots_dir='./plots', savefig=False, with_projections=True):
    for i in range(predictions.shape[0]):
        turn = int(minmax_normalize_param(turn_normalized[i], 0, 1, target_range=(1, 298)))
        # turn = int(unnormalizeTurn(turn_normalized[i]))
        if with_projections:
            f, ax = plt.subplots(2, 3)
        else:
            f, ax = plt.subplots(1, 3)
        ax = np.ravel(ax)
        ax[0].imshow(T_image[i, :, :, 0], cmap='jet')
        ax[0].set_title('T prof')
        ax[0].set_xticks([], [])
        ax[0].set_yticks([], [])
        ax[1].imshow(PS_image[i, :, :, 0], cmap='jet')
        ax[1].set_title('PS @ {}'.format(turn))
        ax[1].set_xticks([], [])
        ax[1].set_yticks([], [])

        ax[2].imshow(predictions[i, :, :, 0], cmap='jet')
        ax[2].set_title('PRED @ {}'.format(turn))
        ax[2].set_xticks([], [])
        ax[2].set_yticks([], [])

        if with_projections:
            ax[4].plot(np.sum(PS_image[i, :, :, 0], 0), label='True')
            ax[4].plot(np.sum(predictions[i, :, :, 0], 0), label='Pred')
            ax[4].legend(loc='center right')
            ax[4].set_title('Time Projection')
            ax[4].set_yticks([], [])
            ax[4].set_xticks([], [])

            ax[5].plot(np.sum(PS_image[i, :, :, 0], 1), label='True')
            ax[5].plot(np.sum(predictions[i, :, :, 0], 1), label='Pred')
            ax[5].legend(loc='center right')
            ax[5].set_title('Energy Projection')
            ax[5].set_yticks([], [])
            ax[5].set_xticks([], [])

            f.delaxes(ax[3])

        plt.tight_layout()
        if savefig:
            plt.savefig(os.path.join(
                plots_dir, f'assess_model_turn{turn}_projections{with_projections}.jpg'), dpi=100, bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def assess_decoder(predictions, turn_normalized, PS_image,
                   plots_dir='./plots', savefig=False):

    for i in range(predictions.shape[0]):
        turn = int(minmax_normalize_param(turn_normalized[i], 0, 1, target_range=(1, 298)))
        # turn = int(unnormalizeTurn(turn_normalized[i]))
        f, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(PS_image[i, :, :, 0], cmap='jet')
        ax[0, 0].set_title('PS @ {}'.format(turn))
        ax[0, 1].imshow(predictions[i, :, :, 0], cmap='jet')
        ax[0, 1].set_title('PREDICTION @ {}'.format(turn))
        ax[1, 0].plot(np.sum(PS_image[i, :, :, 0], 0), label='Target')
        ax[1, 0].plot(np.sum(predictions[i, :, :, 0], 0), label='Prediction')
        ax[1, 0].legend()
        ax[1, 0].set_title('Time Projection')
        ax[1, 1].plot(np.sum(PS_image[i, :, :, 0], 1), label='Target')
        ax[1, 1].plot(np.sum(predictions[i, :, :, 0], 1), label='Prediction')
        ax[1, 1].legend()
        ax[1, 1].set_title('Energy Projection')
        plt.tight_layout()
        if savefig:
            plt.savefig(os.path.join(
                plots_dir, f'assess_decoder_turn{turn}.jpg'), dpi=400)
        else:
            plt.show()
        plt.close()


# def getTimeProfiles_FromData(fname, Ib):
#     with open(fname, 'rb') as f:
#         timeScale_for_tomo = np.load(f)
#         BunchProfiles = np.load(f)
#     BunchProfiles = BunchProfiles*Ib/np.sum(BunchProfiles[:, 0])
#     return timeScale_for_tomo, BunchProfiles

# def getTimeProfiles_FromData_2(fname, Ib):
#     with hp.File(fname, 'r') as sf:
#         BunchProfiles = np.array(sf['bunchProfiles'])
#         EnergyProfiles = np.array(sf['energyProfiles'])
#         phaseSpace_density_array = np.array(sf['phaseSpace_density_array'])
#         x_bin_center_array = np.array(sf['x_bin_center_array'])
#         y_bin_center_array = np.array(sf['y_bin_center_array'])
#     with open(fname, 'rb') as f:
#         timeScale_for_tomo = np.load(f)
#         BunchProfiles = np.load(f)
#     BunchProfiles = BunchProfiles*Ib/np.sum(BunchProfiles[:, 0])
#     return timeScale_for_tomo, BunchProfiles


# def getTimgForModelFromDataFile(fname, Ib, T_normFactor, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, centroid_offset=0):
#     timeScale_for_tomo, BunchProfiles = getTimeProfiles_FromData(fname, Ib)
#     BunchProfiles = BunchProfiles/T_normFactor
#     sel_turns = np.arange(start_turn, skipturns *
#                           (IMG_OUTPUT_SIZE-2*zeropad), skipturns).astype(np.int32)
#     T_img = np.pad(BunchProfiles[:, sel_turns], ((zeropad-centroid_offset, zeropad +
#                                                   centroid_offset), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
#     T_img_ForModel = normalizeIMG(np.reshape(T_img, T_img.shape+(1,)))
#     return T_img_ForModel

def getTimeProfiles_FromData_new(fname, Ib):
    with open(fname, 'rb') as f:
        timeScale_for_tomo = np.load(f)
        BunchProfiles = np.load(f)
    BunchProfiles = BunchProfiles*Ib
    return timeScale_for_tomo, BunchProfiles

def getTimgForModelFromDataFile_new(fname, Ib=1.0, T_normFactor=1.0, IMG_OUTPUT_SIZE=128, zeropad=14, start_turn=1, skipturns=3, centroid_offset=0):
    timeScale_for_tomo, BunchProfiles = getTimeProfiles_FromData_new(fname, Ib)
    BunchProfiles = BunchProfiles/T_normFactor
    sel_turns = np.arange(start_turn, skipturns *
                          (IMG_OUTPUT_SIZE-2*zeropad), skipturns).astype(np.int32)
    # print(sel_turns[0], sel_turns[-1])
    T_img = np.pad(BunchProfiles[:, sel_turns], ((zeropad-centroid_offset, zeropad +
                                                  centroid_offset), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
    T_img_ForModel = minMaxScaleIMG(np.reshape(T_img, T_img.shape+(1,)))

    BunchProfiles = np.pad(BunchProfiles[:, start_turn:sel_turns[-1]+1], ((zeropad-centroid_offset, zeropad +
                                                  centroid_offset), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
    BunchProfiles = minMaxScaleIMG(np.reshape(BunchProfiles, BunchProfiles.shape+(1,)))

    return T_img_ForModel, BunchProfiles

def real_files_to_tensors(data_dir, Ib=1.0, T_normFactor=1.0, IMG_OUTPUT_SIZE=128, zeropad=14, start_turn=1, skipturns=3, centroid_offset=0):
    files = os.listdir(data_dir)
    waterfall_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    bunch_profiles_arr = np.zeros((len(files), 128, 326, 1), dtype=np.float32)

    file_list = []
    # for i, file in enumerate(files):
    i = 0
    for file in files:
        fname = os.path.join(data_dir, file)
        try:
            waterfall, bunch_profiles = getTimgForModelFromDataFile_new(fname)
        except Exception as e:
            print(f'Skipping file {fname}, ', e)
            continue
        waterfall_arr[i] = waterfall
        bunch_profiles_arr[i] = bunch_profiles
        i+=1
        file_list.append(file.split('.npy')[0])
        
    return waterfall_arr[:i], file_list, bunch_profiles_arr[:i]
    
def fwhm(x, y, level=0.5):
    offset_level = np.mean(y[0:10])
    amp = np.max(y) - offset_level
    t1, t2 = interp_f(x, y, level)
    mu = (t1+t2)/2.0
    sigma = (t2-t1)/2.35482
    popt = (mu, sigma, amp)

    return popt


def interp_f(time, bunch, level):
    bunch_th = level*bunch.max()
    time_bet_points = time[1]-time[0]
    taux = np.where(bunch >= bunch_th)
    taux1, taux2 = taux[0][0], taux[0][-1]
    t1 = time[taux1] - (bunch[taux1]-bunch_th) / \
        (bunch[taux1]-bunch[taux1-1]) * time_bet_points
    t2 = time[taux2] + (bunch[taux2]-bunch_th) / \
        (bunch[taux2]-bunch[taux2+1]) * time_bet_points

    return t1, t2


def qgaussian_pdf(x, mu, amp, q, beta):
    y = amp*((1-beta*(1-q)*(x-mu)**2)**(1/(1-q)))
    return np.nan_to_num(y)


def binomial_pdf(x, mu, amp, m, tau_2):
    yy = (1-((x-mu)/tau_2)**2)
    yy[yy < 0] = 0
    y = amp*yy**(m+0.5)  # + offset
    return np.nan_to_num(y)


def symmetrizeProfile(x, y):
    peak_ind = np.argmax(y)
    y_new = np.append(y[:peak_ind], y[peak_ind::-1])
    x_new = np.append(x[:peak_ind], x[:peak_ind+1]+x[peak_ind])
    y_sym = np.interp(x, x_new, y_new)
    return y_sym


def filterFileNamesInFolder(data_path, filter_text=['']):
    fileNames = os.listdir(data_path)
    fNames = []
    for f in fileNames:
        #        for filter_text in filter_texts:
        if filter_text in f:
            fNames.append(f)

    return fNames


def running_mean(x, N):

    if np.ndim(x) == 2:
        moving_average = np.zeros(np.shape(x))
        for i in np.arange(np.shape(x)[1]):
            moving_average[:, i] = np.convolve(
                x[:, i], np.ones((N,))/N, mode='same')
    else:
        moving_average = np.convolve(x, np.ones((N,))/N, mode='same')

    return moving_average


def window_mean(x, W, axis=0, zeropad=14):
    moving_average = np.copy(x)
    if axis == 0:
        for i in np.arange(np.shape(moving_average)[0]):
            moving_average[i, :zeropad] = moving_average[i, zeropad] * np.ones(zeropad) 
            moving_average[i, -zeropad:] = moving_average[i, -zeropad-1] * np.ones(zeropad)
            moving_average[i,:] = np.convolve(moving_average[i], np.ones((W,))/W, mode='same')
        moving_average[:, :zeropad] = x[:, :zeropad]
        moving_average[:, -zeropad:] = x[:, -zeropad:]
    elif axis == 1:
        for i in np.arange(np.shape(moving_average)[1]):
            moving_average[:zeropad, i] = moving_average[zeropad, i] * np.ones(zeropad) 
            moving_average[-zeropad:, i] = moving_average[-zeropad-1, i] * np.ones(zeropad)
            moving_average[:, i] = np.convolve(moving_average[:, i], np.ones((W,))/W, mode='same')
        moving_average[:zeropad, :] = x[:zeropad, :]
        moving_average[-zeropad:, :] = x[-zeropad:, :]
    return moving_average

def conv2D_output_size(input_size, out_channels, padding, kernel_size, stride,
                       dilation=None):
    if dilation is None:
        dilation = (1, ) * 2
    if isinstance(padding, int):
        padding = (padding, ) * 2
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, ) * 2
    if isinstance(stride, int):
        stride = (stride, ) * 2

    output_size = (
        np.floor((input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) /
                 stride[0] + 1).astype(int),
        np.floor((input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) /
                 stride[1] + 1).astype(int),
        out_channels,
    )
    return output_size
