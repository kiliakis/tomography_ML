import numpy as np
import pickle as pk
import h5py as hp
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import re
import glob
import bisect
from scipy import signal
from scipy.interpolate import interp1d
from tensorflow.keras.models import load_model


def plot_sample(x_valid, samples, autoenc, figname=None):

    ncols = len(samples)
    # Get nrows * nrows random images
    # sample = np.random.choice(np.arange(len(x_train)),
    #                         size=ncols, replace=False)

    samples_X = tf.gather(x_valid, samples)
    pred_samples_X = autoenc.predict(samples_X)
    # samples_y = tf.gather(y_train, sample)

    # Create 3x3 grid of figures
    fig, axes = plt.subplots(nrows=2, ncols=ncols, figsize=(14, 7))
    # axes = np.ravel(axes)
    for i in range(ncols):
        sample_X = samples_X[i, 14:-14, 14:-14].numpy()
        pred_sample_X = pred_samples_X[i, 14:-14, 14:-14].numpy()
        ax = axes[0, i]
        ax.set_xticks([])
        ax.set_yticks([])
        # show the image
        ax.imshow(sample_X, cmap='jet')
        # Set the label
        ax.set_title(f'Real')

        ax = axes[1, i]
        ax.set_xticks([])
        ax.set_yticks([])
        # show the image
        ax.imshow(pred_sample_X, cmap='jet')
        # Set the label
        ax.set_title(
            f'Pred, MAE: {np.mean(np.abs(sample_X - pred_sample_X)):.2e}')

    if figname is not None:
        plt.savefig(figname, dpi=300)
    else:
        plt.show()
    plt.close()


def get_best_model_timestamp(path, model='enc'):
    from sort_trial_summaries import extract_trials
    header, rows = extract_trials(path)
    for row in rows:
        if model in row[header.index('model')]:
            return row[header.index('date')]


def visualize_weights(model_filename, plots_dir, prefix=''):
    model = load_model(model_filename)

    weights_per_layer = {}
    for layer in model.layers:
        if len(layer.get_weights()) > 0:
            # [0] for weights, [1] for biases
            weights_per_layer[layer.name] = layer.get_weights()[0]

    for layer_name, weights in weights_per_layer.items():
        plt.figure()
        plt.hist(weights.flatten(), bins=50, range=(-0.3, 0.3))
        plt.title(f'Weight Distribution for layer {layer_name}')
        plt.xlabel('Weight Value, Total weights = {}'.format(weights.size))
        plt.ylabel('Frequency')
        # plt.show()
        plt.savefig(os.path.join(plots_dir, f'weights_{prefix}_{layer_name}.jpg'), dpi=400)
        plt.close()


def plot_feature_extractor_evaluation(latent_pred, latent_true, normalization, figname=None):
    # unormalized latent space
    latent_unnorm = unnormalize_params(
        latent_true[:, 0], latent_true[:, 1], latent_true[:, 2],
        latent_true[:, 3], latent_true[:, 4], latent_true[:, 5],
        latent_true[:, 6], normalization=normalization)
    latent_unnorm = np.array(latent_unnorm).T

    # unormalized predicted latent space
    latent_pred_unnorm = unnormalize_params(
        latent_pred[:, 0], latent_pred[:, 1], latent_pred[:, 2],
        latent_pred[:, 3], latent_pred[:, 4], latent_pred[:, 5],
        latent_pred[:, 6], normalization=normalization)
    latent_pred_unnorm = np.array(latent_pred_unnorm).T

    # absolute difference
    diffs = np.abs(latent_unnorm - latent_pred_unnorm)

    # Encoder, graphical evaluation
    evaluation_config = {
        0: {'xlabel': 'Phase Error [deg]',
            'range': (0, 5),
            'xticks': np.arange(0, 5.1, 0.5),
            'desired': 2,
            'multiplier': 1},
        1: {'xlabel': 'Energy Error [MeV]',
            'range': (0, 5),
            'xticks': np.arange(0, 5.1, 0.5),
            'desired': 2,
            'multiplier': 1},
        2: {'xlabel': 'Bunch Length [ps]',
            'range': (0, 50),
            'xticks': np.arange(0, 50.5, 5),
            'desired': 25,
            'multiplier': 1e12},
        3: {'xlabel': 'Bunch Intensity [1e9]',
            'range': (0, 5),
            'xticks': np.arange(0, 5.1, 0.5),
            'desired': 1.5,
            'multiplier': 1e-9},
        4: {'xlabel': 'V_rf [MV]',
            'range': (0, 0.3),
            'xticks': np.arange(0, 0.31, 0.05),
            'desired': 0.1,
            'multiplier': 1},
        5: {'xlabel': 'mu [a.u.]',
            'range': (0, 0.6),
            'xticks': np.arange(0, 0.61, 0.1),
            'desired': 0.2,
            'multiplier': 1},
        6: {'xlabel': 'V_rf SPS [MV]',
            'range': (0, 0.3),
            'xticks': np.arange(0, 0.31, 0.05),
            'desired': 0.1,
            'multiplier': 1},
    }

    fig, axes = plt.subplots(ncols=2, nrows=4, sharex=False,
                            sharey=False, figsize=(8, 8))
    axes = np.ravel(axes, order='F')

    for idx, ax in enumerate(axes):
        if idx == 7:
            break
        plt.sca(ax)
        config = evaluation_config[idx]
        hist, bins, _ = plt.hist(
            diffs[:, idx]*config['multiplier'], bins=50, range=config['range'], label='samples')
        cumsum = np.cumsum(hist) / diffs.shape[0]
        b = bisect.bisect(cumsum, 0.95)
        if b+1 < len(bins):
            x = bins[b+1]
        else:
            x = bins[-1]
        plt.axvline(x=x, color='tab:orange',
                    label=f'95% < {x:.2f}')
        plt.xticks(config['xticks'])
        plt.xlabel(config['xlabel'])
        if idx < 4:
            plt.ylabel('No. Samples')
        plt.axvline(x=config['desired'], color='black',
                    label=f'Desired < {config["desired"]:.1f}')
        plt.legend(loc='upper right')
        plt.gca().set_facecolor('0.85')

    # delete last
    fig.delaxes(axes[-1])

    plt.tight_layout()
    plt.savefig(figname, dpi=400, bbox_inches='tight')
    plt.close()


def plot_multi_loss(lines, title='', figname=None):
    nrows = len(lines)//2
    ncols = 2
    nrows = (nrows+ncols-1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5*ncols, 3*nrows), sharex=True)
    axes = np.ravel(axes)
    fig.suptitle(title)
    line_to_ax = {}
    i = 0
    for line in lines.keys():
        key = line.lower().replace('_val', '')
        if key not in line_to_ax:
            line_to_ax[key] = axes[i]
            i += 1
        ax = line_to_ax[key]
        if 'val' in line.lower():
            marker = 'x'
        else:
            marker = '.'
        plt.sca(ax)
        plt.semilogy(lines[line], marker=marker, label=line)
        plt.grid(True)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(ncol=2)
        plt.tight_layout()

    if figname:
        plt.savefig(figname, dpi=300)
        plt.close()


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

def get_model_size(model):
    # Count the total parameters
    total_parameters = model.count_params()
    
    # Assuming 32-bit floats (4 bytes) per parameter
    total_bytes = total_parameters * 4
    
    # Convert bytes to megabytes
    total_megabytes = total_bytes / (1024 * 1024)
    
    return total_megabytes


def normalizeIMG(img, maxPixel=1):
    return (img / (maxPixel/2)) - 1


def unnormalizeIMG(img, maxPixel=1):
    return (img+1)*(maxPixel/2)


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
    bin_centers = np.linspace(cut_left, cut_right, n_slices, endpoint=False)
    # bin_centers = (edges[:-1] + edges[1:])/2
    return bin_centers


def loadTF(path, beam=2, cut=100):
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

        # This is equal to sf['columns'][0][3]
        n_macroparticles = BunchProfiles[:, 0].sum()
        
        # fig = plt.figure(1)
        # offset = 1000
        # for turn in np.arange(0, BunchProfiles.shape[1], 5):
        #     plt.plot(time_scale, turn*offset + BunchProfiles[:, turn])
        # plt.xlabel('Time [s]')
        # plt.yticks([], [])
        # plt.ylabel('Time profiles')
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig('plots/input_mountain_range.jpg', dpi=400)
        # plt.close()

        # fig = plt.figure(2)
        # plt.imshow(BunchProfiles[:, :300].T, cmap='jet', aspect='auto', origin='lower')
        # plt.xlabel('Bins')
        # plt.ylabel('Turns')
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig('plots/input_waterfall.jpg', dpi=400)
        # plt.close()

        # plt.plot(time_scale, BunchProfiles[:, 0], label='before_tf')
        if (time_scale is not None) and (freq_array is not None) and (TF_array is not None):
            _, BunchProfiles = bunchProfile_TFconvolve(BunchProfiles, time_scale,
                                                       freq_array, TF_array)
            
            # fig = plt.figure(3)
            # offset = 1000
            # for turn in np.arange(0, BunchProfiles.shape[1], 5):
            #     plt.plot(time_scale, turn*offset + BunchProfiles[:, turn])
            # plt.xlabel('Time [s]')
            # plt.yticks([], [])
            # plt.ylabel('Time profiles')
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig('plots/input_mountain_range_with_tf.jpg', dpi=400)
            # plt.close()


            # fig = plt.figure(4)
            # plt.imshow(BunchProfiles[:, :300].T, cmap='jet', aspect='auto', origin='lower')
            # plt.xlabel('Time')
            # plt.ylabel('Turns')
            # plt.tight_layout()
            # # plt.show()
            # plt.savefig('plots/input_waterfall_with_tf.jpg', dpi=400)
            # plt.close()
            # plt.figure(2)
            # plt.plot(freq_array, np.abs(TF_array))
            # plt.figure(5)
            # plt.plot(time_scale, BunchProfiles[:, 0], label='after_tf')

        # BunchProfiles = BunchProfiles / sf['columns'][0][3]*paramsDict['int']
        BunchProfiles = BunchProfiles / n_macroparticles

        # plt.legend()
        # plt.show()
        # plt.savefig('plots/profile_before_after_tf.jpg', dpi=400)
        # plt.close()

        # EnergyProfiles = np.array(sf['energyProfiles'])/n_macroparticles*paramsDict['int']
        EnergyProfiles = np.array(sf['energyProfiles']) / n_macroparticles

        phaseSpace_density_array = np.array(sf['phaseSpace_density_array'])
        # PS_imgs = np.zeros((IMG_OUTPUT_SIZE, IMG_OUTPUT_SIZE,
        #                    phaseSpace_density_array.shape[1]))
        
        # This will transform from 99801x501 to 128x128x501
        PS_imgs = np.pad(phaseSpace_density_array.reshape((99, 99, -1)).transpose(1,0,2)/n_macroparticles, 
                         ((zeropad, zeropad+1), (zeropad, zeropad+1), (0,0)))
        # for i in range(phaseSpace_density_array.shape[1]):
        #     # turn_PS = np.transpose(np.reshape(
        #     #     phaseSpace_density_array[:, i], (99, 99)))/n_macroparticles*paramsDict['int']
        #     turn_PS = np.transpose(np.reshape(
        #         phaseSpace_density_array[:, i], (99, 99)))/n_macroparticles
        #     PS_imgs[:, :, i] = np.pad(
        #         turn_PS, ((zeropad, zeropad+1), (zeropad, zeropad+1)))
        sel_turns = np.arange(
            start_turn, skipturns*(IMG_OUTPUT_SIZE-2*zeropad), skipturns).astype(np.int32)
        PS_img_dec = PS_imgs[:, :, sel_turns]
        E_img = np.pad(EnergyProfiles[:, sel_turns], ((
            zeropad, zeropad), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
        T_img = np.pad(BunchProfiles[:, sel_turns], ((
            zeropad, zeropad), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
        
        # fig = plt.figure(5)
        # plt.imshow(T_img.T, cmap='jet', aspect='auto', origin='lower')
        # plt.xlabel('Time')
        # plt.ylabel('Turns')
        # plt.tight_layout()
        # # plt.show()
        # plt.savefig('plots/input_waterfall_with_tf_formatted.jpg', dpi=400)
        # plt.close()
    return paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec


def read_pk(fname):
    # Returns a random sample of percent filenames from input path
    data = pk.loads(tf.io.decode_raw(tf.io.read_file(fname), tf.uint8))
    return data['turn'], data['T_img'], data['PS'], data['fn'], data['params']


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


def fast_tensor_load(path, percent=1.0, max_files=-1, dtype='float32'):
    x_train, y_train = [], []
    all_files = glob.glob(path)
    if max_files > 0:
        all_files = all_files[max_files]
    # For every file that matches the regexp
    for file in all_files:
        print(f'Loading {file}')
        # decompress and load file
        with np.load(file) as data:
            x, y = data['x'], data['y']
        # Keep a smaller percentage if needed
        if percent < 1 and percent > 0:
            points = len(y)
            keep_points = np.random.choice(
                points, int(points * percent), replace=False)
            x, y = x[keep_points], y[keep_points]
        # append to list
        x_train.append(tf.convert_to_tensor(x, dtype=dtype))
        y_train.append(tf.convert_to_tensor(y, dtype=dtype))
    # make the final tensor
    x_train = tf.concat(x_train, axis=0)
    y_train = tf.concat(y_train, axis=0)
    return x_train, y_train



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


def load_encoder_data(pk_file, normalization, normalize=True, img_normalize='default'):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    # PS = np.reshape(PS, PS.shape+(1,))
    # turn_num = normalizeTurn(turn_num)
    if img_normalize == 'default':
        T_img = normalizeIMG(T_img)
    elif img_normalize == 'minmax':
        T_img = minMaxScaleIMG(T_img)
    elif img_normalize == 'off':
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

#     T_img = normalizeIMG(T_img)
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
        if len(keep_idx) == 0:
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
    for file in all_files:
        print(f'Loading {file}')
        # decompress and load file
        with np.load(file) as data:
            wf, turn, latent, ps = data['WFs'], data['turns'], data['latents'], data['PSs']
        # Keep a smaller percentage if needed
        if percent < 1 and percent > 0:
            points = len(wf)
            keep_points = np.random.choice(
                points, int(points * percent), replace=False)
            wf, turn, latent, ps = wf[keep_points], turn[keep_points], latent[keep_points], ps[keep_points]
        # append to list
        wf_arr.append(wf)
        turn_arr.append(turn)
        latent_arr.append(latent)
        ps_arr.append(ps)

    # make the final tensor
    wf_arr = tf.convert_to_tensor(np.concatenate(wf_arr, axis=0))
    turn_arr = tf.convert_to_tensor(np.concatenate(turn_arr, axis=0))
    latent_arr = tf.convert_to_tensor(np.concatenate(latent_arr, axis=0))
    ps_arr = tf.convert_to_tensor(np.concatenate(ps_arr, axis=0))

    return wf_arr, turn_arr, latent_arr, ps_arr

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
        i += 1
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
        i += 1
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
        i += 1
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
        turn = int(minmax_normalize_param(
            turn_normalized[i], 0, 1, target_range=(1, 298)))
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
        turn = int(minmax_normalize_param(
            turn_normalized[i], 0, 1, target_range=(1, 298)))
        # turn = int(unnormalizeTurn(turn_normalized[i]))
        f, ax = plt.subplots(2, 2)

        ax[0, 0].imshow(PS_image[i, :, :, 0], cmap='jet')
        ax[0, 0].set_title('PS @ {}'.format(turn))
        ax[0, 1].imshow(predictions[i, :, :, 0], cmap='jet')
        ax[0, 1].set_title('PREDICTION @ {}'.format(turn))
        ax[1, 0].plot(np.sum(PS_image[i, :, :, 0], 0), label='Target')
        ax[1, 0].plot(np.sum(predictions[i, :, :, 0], 0), label='Prediction')
        ax[1, 0].legend(loc='upper left')
        ax[1, 0].set_title('Time Projection')
        ax[1, 0].set_yticks([], [])
        ax[1, 0].set_xticks([], [])

        ax[1, 1].plot(np.sum(PS_image[i, :, :, 0], 1), label='Target')
        ax[1, 1].plot(np.sum(predictions[i, :, :, 0], 1), label='Prediction')
        ax[1, 1].legend(loc='upper left')
        ax[1, 1].set_title('Energy Projection')
        ax[1, 1].set_yticks([], [])
        ax[1, 1].set_xticks([], [])

        plt.tight_layout()
        if savefig:
            plt.savefig(os.path.join(
                plots_dir, f'assess_decoder_turn{turn}.jpg'), dpi=100,
                bbox_inches='tight')
        else:
            plt.show()
        plt.close()


def correctTriggerOffsets(x, frames, triggerOffsets):

    NTurns = np.shape(frames)[1]
    frames_new = np.zeros(np.shape(frames))
    for i in np.arange(NTurns):
        x_temp = x + triggerOffsets[i]
        frame = frames[:, i]
        extrap_value = np.mean(frame[0:10])
        frame_interp = interp1d(
            x_temp, frame, bounds_error=False, fill_value=extrap_value)
        frames_new[:, i] = frame_interp(x)

    return frames_new


def getTriggerOffset(BunchProfiles, filter_n=12):
    # import matplotlib.pyplot as plt
    dataPoints = np.shape(BunchProfiles)[0]
    mass_centre = (BunchProfiles.T @ np.arange(dataPoints)) / \
        np.sum(BunchProfiles, axis=0)
    filtered = signal.filtfilt(np.ones(filter_n) / filter_n, 1, mass_centre)
    return mass_centre - filtered


def correctForTriggerOffset(timeScale, singleBunchFrame, filter_n=12,  iterations=1):
    # import matplotlib.pyplot as plt
    dt = timeScale[1] - timeScale[0]
    singleBunchFrame_iter = singleBunchFrame
    try:
        for i in np.arange(iterations):
            trigger_offsets = getTriggerOffset(
                singleBunchFrame_iter, filter_n=filter_n)
            trigger_offsets *= dt
            singleBunchFrame_iter = correctTriggerOffsets(
                timeScale, singleBunchFrame_iter, -trigger_offsets)
        singleBunchFrame_new = singleBunchFrame_iter
    except Exception as e:
        print(e)
        print("No filtering of the scope trigger offsets")
        singleBunchFrame_new = singleBunchFrame

    # dataPoints = np.shape(singleBunchFrame_new)[0]
    # mass_centre = (singleBunchFrame_new.T @ np.arange(dataPoints)) / np.sum(singleBunchFrame_new, axis=0)

    return singleBunchFrame_new


def getTimeProfiles_FromData_new(fname, Ib):
    with open(fname, 'rb') as f:
        timeScale_for_tomo = np.load(f)
        BunchProfiles = np.load(f)
    # divide the profiles by the integral, multiply by intensity
    BunchProfiles = BunchProfiles / np.sum(BunchProfiles[:, 0], axis=0) * Ib
    return timeScale_for_tomo, BunchProfiles


def getTimgForModelFromDataFile_new(fname, Ib=1.0, T_normFactor=1.0, IMG_OUTPUT_SIZE=128,
                                    zeropad=14, start_turn=1, skipturns=3,
                                    centroid_offset=0, corrTriggerOffset=False,
                                    filter_n=12, iterations=1):
    timeScale_for_tomo, BunchProfiles = getTimeProfiles_FromData_new(fname, Ib)
    # print(timeScale_for_tomo)
    # Apply correction
    if corrTriggerOffset == True:
        # print(BunchProfiles.shape)
        BunchProfiles = correctForTriggerOffset(timeScale_for_tomo, BunchProfiles,
                                                filter_n=filter_n, iterations=iterations)

    # finally normalize with the T_normFactor (max intensity per slice)
    BunchProfiles = BunchProfiles / T_normFactor
    sel_turns = np.arange(start_turn, skipturns *
                          (IMG_OUTPUT_SIZE-2*zeropad), skipturns).astype(np.int32)
    # print(sel_turns[0], sel_turns[-1])
    T_img = np.pad(BunchProfiles[:, sel_turns], ((zeropad-centroid_offset, zeropad +
                                                  centroid_offset), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
    T_img_ForModel = np.reshape(T_img, T_img.shape+(1,))
    # T_img_ForModel = minMaxScaleIMG(np.reshape(T_img, T_img.shape+(1,)))

    BunchProfiles = np.pad(BunchProfiles[:, start_turn:sel_turns[-1]+1], ((zeropad-centroid_offset, zeropad +
                                                                           centroid_offset), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
    BunchProfiles = np.reshape(BunchProfiles, BunchProfiles.shape+(1,))
    # BunchProfiles = minMaxScaleIMG(np.reshape(BunchProfiles, BunchProfiles.shape+(1,)))

    return T_img_ForModel, BunchProfiles


def real_files_to_tensors(data_dir, Ib=1.0, T_normFactor=1.0, IMG_OUTPUT_SIZE=128,
                          zeropad=14, start_turn=1, skipturns=3, centroid_offset=0,
                          corrTriggerOffset=False, filter_n=12, iterations=1):
    files = os.listdir(data_dir)
    waterfall_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    bunch_profiles_arr = np.zeros((len(files), 128, 326, 1), dtype=np.float32)

    file_list = []
    # for i, file in enumerate(files):
    i = 0
    for file in files:
        fname = os.path.join(data_dir, file)
        try:
            waterfall, bunch_profiles = getTimgForModelFromDataFile_new(fname,
                                                                        Ib=Ib,
                                                                        T_normFactor=T_normFactor,
                                                                        corrTriggerOffset=corrTriggerOffset,
                                                                        filter_n=filter_n,
                                                                        iterations=iterations)
        except Exception as e:
            print(f'Skipping file {fname}, ', e)
            continue
        waterfall_arr[i] = waterfall
        bunch_profiles_arr[i] = bunch_profiles
        i += 1
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
            moving_average[i, :zeropad] = moving_average[i,
                                                         zeropad] * np.ones(zeropad)
            moving_average[i, -zeropad:] = moving_average[i, -
                                                          zeropad-1] * np.ones(zeropad)
            moving_average[i, :] = np.convolve(
                moving_average[i], np.ones((W,))/W, mode='same')
        moving_average[:, :zeropad] = x[:, :zeropad]
        moving_average[:, -zeropad:] = x[:, -zeropad:]
    elif axis == 1:
        for i in np.arange(np.shape(moving_average)[1]):
            moving_average[:zeropad,
                           i] = moving_average[zeropad, i] * np.ones(zeropad)
            moving_average[-zeropad:,
                           i] = moving_average[-zeropad-1, i] * np.ones(zeropad)
            moving_average[:, i] = np.convolve(
                moving_average[:, i], np.ones((W,))/W, mode='same')
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




