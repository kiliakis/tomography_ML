import numpy as np
import pickle as pk
import h5py as hp
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import re


def plot_loss(lines, title='', figname=None):
    plt.figure()
    plt.title(title)
    for line in lines.keys():
        plt.semilogy(lines[line], marker='x', label=line)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    if figname:
        plt.savefig(figname, dpi=300)
    plt.close()

def get_cmap(path=''):

    if path == '':
        path = os.path.realpath(
            'C:\\Users\\teoarg\\cernbox\\python\\myFunctions')

    cmap_white_blue_red_import = np.loadtxt(
        os.path.join(path, 'colormap\\colormap.txt'))
    cmap_white_blue_red_interparray = np.array(np.linspace(
        0., 1., len(cmap_white_blue_red_import[:, 0])), ndmin=2)
    cmap_white_blue_red_red = np.hstack((cmap_white_blue_red_interparray.T, np.array(
        cmap_white_blue_red_import[:, 0], ndmin=2).T, np.array(cmap_white_blue_red_import[:, 0], ndmin=2).T))
    cmap_white_blue_red_green = np.hstack((cmap_white_blue_red_interparray.T, np.array(
        cmap_white_blue_red_import[:, 1], ndmin=2).T, np.array(cmap_white_blue_red_import[:, 1], ndmin=2).T))
    cmap_white_blue_red_blue = np.hstack((cmap_white_blue_red_interparray.T, np.array(
        cmap_white_blue_red_import[:, 2], ndmin=2).T, np.array(cmap_white_blue_red_import[:, 2], ndmin=2).T))

    cmap_white_blue_red_red = tuple(map(tuple, cmap_white_blue_red_red))
    cmap_white_blue_red_green = tuple(map(tuple, cmap_white_blue_red_green))
    cmap_white_blue_red_blue = tuple(map(tuple, cmap_white_blue_red_blue))

    cdict_white_blue_red = {'red': cmap_white_blue_red_red,
                            'green': cmap_white_blue_red_green, 'blue': cmap_white_blue_red_blue}

    cmap_white_blue_red = LinearSegmentedColormap(
        'WhiteBlueRed', cdict_white_blue_red)

    return cmap_white_blue_red


def running_mean(x, N):

    if np.ndim(x) == 2:
        moving_average = np.zeros(np.shape(x))
        for i in np.arange(np.shape(x)[1]):
            moving_average[:, i] = np.convolve(
                x[:, i], np.ones((N,))/N, mode='same')
    else:
        moving_average = np.convolve(x, np.ones((N,))/N, mode='same')

    return moving_average


def extract_data_Fromfolder(fn, simulations_dir, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3):

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
        BunchProfiles = np.array(sf['bunchProfiles']) / \
            sf['columns'][0][3]*paramsDict['int']
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


def normalizeIMG(img, maxPixel=1):
    return (img / (maxPixel/2)) - 1


def unnormalizeIMG(img, maxPixel=1):
    return (img+1)*(maxPixel/2)


def normalizeTurn(turn_num, maxTurns=300.0):
    return (turn_num / (maxTurns/2)) - 1


def unnormalizeTurn(turn_num, maxTurns=300.0):
    return (turn_num+1)*(maxTurns/2)


def load_encoder_data(pk_file):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    # PS = np.reshape(PS, PS.shape+(1,))
    # turn_num = normalizeTurn(turn_num)
    T_img = normalizeIMG(T_img)
    # PS = normalizeIMG(PS)
    phEr = float(params_dict['phEr'])
    enEr = float(params_dict['enEr'])
    bl = float(params_dict['bl'])
    inten = float(params_dict['int'])
    Vrf = float(params_dict['Vrf'])
    mu = float(params_dict['mu'])
    VrfSPS = float(params_dict['VrfSPS'])
    phEr, enEr, bl, inten, Vrf, mu, VrfSPS  = normalize_params(phEr, enEr, bl, inten, Vrf, mu, VrfSPS)
    # T_normFactor = float(params_dict['T_normFactor'])
    # B_normFactor = float(params_dict['B_normFactor'])
    # return turn_num, T_img, PS, fn, phEr, enEr, bl, inten, Vrf, mu, VrfSPS, T_normFactor, B_normFactor
    return (T_img, [phEr, enEr, bl, inten, Vrf, mu, VrfSPS])


def load_decoder_data(pk_file):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    # T_img = np.reshape(T_img, T_img.shape+(1,))
    PS = np.reshape(PS, PS.shape+(1,))
    turn_num = normalizeTurn(turn_num)
    # T_img = normalizeIMG(T_img)
    PS = normalizeIMG(PS)
    phEr = float(params_dict['phEr'])
    enEr = float(params_dict['enEr'])
    bl = float(params_dict['bl'])
    inten = float(params_dict['int'])
    Vrf = float(params_dict['Vrf'])
    mu = float(params_dict['mu'])
    VrfSPS = float(params_dict['VrfSPS'])
    phEr, enEr, bl, inten, Vrf, mu, VrfSPS  = normalize_params(phEr, enEr, bl, inten, Vrf, mu, VrfSPS)

    # T_normFactor = float(params_dict['T_normFactor'])
    # B_normFactor = float(params_dict['B_normFactor'])
    return ([turn_num, phEr, enEr, bl, inten, Vrf, mu, VrfSPS], PS)


def load_model_data_new(pk_file):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    PS = np.reshape(PS, PS.shape+(1,))
    turn_num = normalizeTurn(turn_num)
    T_img = normalizeIMG(T_img)
    PS = normalizeIMG(PS)
    phEr = float(params_dict['phEr'])
    enEr = float(params_dict['enEr'])
    bl = float(params_dict['bl'])
    inten = float(params_dict['int'])
    Vrf = float(params_dict['Vrf'])
    mu = float(params_dict['mu'])
    VrfSPS = float(params_dict['VrfSPS'])
    T_normFactor = float(params_dict['T_normFactor'])
    B_normFactor = float(params_dict['B_normFactor'])
    return turn_num, T_img, PS, fn, phEr, enEr, bl, inten, Vrf, mu, VrfSPS, T_normFactor, B_normFactor


def load_model_data(pk_file):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    PS = np.reshape(PS, PS.shape+(1,))
    turn_num = normalizeTurn(turn_num)
    T_img = normalizeIMG(T_img)
    PS = normalizeIMG(PS)
    phEr = float(params_dict['phEr'])
    enEr = float(params_dict['enEr'])
    bl = float(params_dict['bl'])
    inten = float(params_dict['int'])
    Vrf = float(params_dict['Vrf'])
    mu = float(params_dict['mu'])
    T_normFactor = float(params_dict['T_normFactor'])
    B_normFactor = float(params_dict['B_normFactor'])
    return turn_num, T_img, PS, fn, phEr, enEr, bl, inten, Vrf, mu, T_normFactor, B_normFactor


def load_model_data_old(pk_file):
    turn_num, T_img, PS, fn, params_dict = read_pk(pk_file)
    T_img = np.reshape(T_img, T_img.shape+(1,))
    PS = np.reshape(PS, PS.shape+(1,))
    turn_num = normalizeTurn(turn_num)
    T_img = normalizeIMG(T_img)
    PS = normalizeIMG(PS)
    phEr = float(params_dict['phEr'])
    enEr = float(params_dict['enEr'])
    bl = float(params_dict['bl'])
    inten = float(params_dict['int'])
    T_normFactor = float(params_dict['T_normFactor'])
    B_normFactor = float(params_dict['B_normFactor'])
    return turn_num, T_img, PS, fn, phEr, enEr, bl, inten, T_normFactor, B_normFactor

def encoder_files_to_tensors(files):
    feature_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    output_arr = np.zeros((len(files), 7), dtype=np.float32)
    for i, file in enumerate(files):
        features, output = load_encoder_data(file)
        feature_arr[i] = features
        output_arr[i] = output
    x_train = tf.convert_to_tensor(feature_arr)
    y_train = tf.convert_to_tensor(output_arr)
    return x_train, y_train

def decoder_files_to_tensors(files):
    feature_arr = np.zeros((len(files), 8), dtype=np.float32)
    output_arr = np.zeros((len(files), 128, 128, 1), dtype=np.float32)
    for i, file in enumerate(files):
        features, output = load_decoder_data(file)
        feature_arr[i] = features
        output_arr[i] = output
    x_train = tf.convert_to_tensor(feature_arr)
    y_train = tf.convert_to_tensor(output_arr)
    return x_train, y_train

def normalize_param(val, mu, sig):
    return (val-mu)/sig


def unnormalize_param(norm_val, mu, sig):
    return norm_val*sig+mu

# TODO: Here add VrfSPS + mu and sig
def normalize_params(phErs, enErs, bls, intens, Vrf, mu, VrfSPS,
                     phEr_mu=0, phEr_sig=50,
                     enEr_mu=0, enEr_sig=100,
                     bl_mu=1.4e-9, bl_sig=0.2e-9,
                     intens_mu=1.225e11, intens_sig=0.37e11,
                     Vrf_mu=6, Vrf_sig=2.2,
                     mu_mu=2, mu_sig=1,
                     VrfSPS_mu=6, VrfSPS_sig=2.2
                     ):
    return normalize_param(phErs, phEr_mu, phEr_sig),\
        normalize_param(enErs, enEr_mu, enEr_sig),\
        normalize_param(bls, bl_mu, bl_sig),\
        normalize_param(intens, intens_mu, intens_sig),\
        normalize_param(Vrf, Vrf_mu, Vrf_sig),\
        normalize_param(mu, mu_mu, mu_sig),\
        normalize_param(VrfSPS, VrfSPS_mu, VrfSPS_sig)

# TODO: Here add VrfSPS + mu and sig
def unnormalize_params(phErs_norm, enErs_norm, bls_norm, intens_norm, Vrf_norm,
                       mu_norm, VrfSPS_norm,
                       phEr_mu=0, phEr_sig=50,
                       enEr_mu=0, enEr_sig=100,
                       bl_mu=1.4e-9, bl_sig=0.2e-9,
                       intens_mu=1.225e11, intens_sig=0.37e11,
                       Vrf_mu=6, Vrf_sig=2.2,
                       mu_mu=2, mu_sig=1,
                       VrfSPS_mu=0, VrfSPS_sig=1
                       ):
    return unnormalize_param(phErs_norm, phEr_mu, phEr_sig),\
        unnormalize_param(enErs_norm, enEr_mu, enEr_sig),\
        unnormalize_param(bls_norm, bl_mu, bl_sig),\
        unnormalize_param(intens_norm, intens_mu, intens_sig),\
        unnormalize_param(Vrf_norm, Vrf_mu, Vrf_sig),\
        unnormalize_param(mu_norm, mu_mu, mu_sig),\
        unnormalize_param(VrfSPS_norm, VrfSPS_mu, VrfSPS_sig)


def assess_model(model, turn_normalized, T_image, PS_image, epoch=None):
    #    predictions, _ = model.predictPS(T_image, turn_normalized, training = False)
    predictions, _ = model.predictPS(T_image, turn_normalized)
    for i in range(predictions.shape[0]):
        turn = unnormalizeTurn(turn_normalized[i])
        f, ax = plt.subplots(2, 3)
        ax[0, 0].imshow(T_image[i, :, :, 0],  vmin=-1, vmax=1)
        ax[0, 0].set_title(
            'T prof' if epoch is None else 'Ep {} T prof'.format(epoch))
        ax[0, 1].imshow(PS_image[i, :, :, 0], vmin=-1, vmax=1)
        ax[0, 1].set_title('Ep {} PS @ {}'.format(epoch, int(turn)))
        ax[0, 2].imshow(predictions[i, :, :, 0], vmin=-1, vmax=1)
        ax[0, 2].set_title('Ep {} PREDICTION @ {}'.format(epoch, int(turn)))
        ax[1, 1].plot(np.sum(PS_image[i, :, :, 0], 0), label='Target')
        ax[1, 1].plot(np.sum(predictions[i, :, :, 0], 0), label='Prediction')
        ax[1, 1].legend()
        ax[1, 1].set_title('Time Projection')
        ax[1, 2].plot(np.sum(PS_image[i, :, :, 0], 1), label='Target')
        ax[1, 2].plot(np.sum(predictions[i, :, :, 0], 1), label='Prediction')
        ax[1, 2].legend()
        ax[1, 2].set_title('Energy Projection')
        f.delaxes(ax[1, 0])
        plt.show()


def assess_decoder(model, turn_normalized, PS_image, phEr, enEr, bl, inten, Vrf,
                   mu, VrfSPS, epoch=None, figname='assess_decoder.png', savefig=False):

    norm_pars = tf.expand_dims(tf.transpose(tf.convert_to_tensor(
        normalize_params(phEr, enEr, bl, inten, Vrf, mu, VrfSPS))), axis=0)
    predictions = model.decode(model.extend(norm_pars, turn_normalized))

    for i in range(predictions.shape[0]):
        turn = unnormalizeTurn(turn_normalized[i])
        f, ax = plt.subplots(2, 2)
        ax[0, 0].imshow(PS_image[i, :, :, 0], vmin=-1, vmax=1)
        ax[0, 0].set_title('Ep {} PS @ {}'.format(epoch, turn))
        ax[0, 1].imshow(predictions[i, :, :, 0], vmin=-1, vmax=1)
        ax[0, 1].set_title('Ep {} PREDICTION @ {}'.format(epoch, turn))
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
            plt.savefig(figname, dpi=300)
        else:
            plt.show()
        plt.close()


def getTimeProfiles_FromData(fname, Ib):
    with open(fname, 'rb') as f:
        timeScale_for_tomo = np.load(f)
        BunchProfiles = np.load(f)
    BunchProfiles = BunchProfiles*Ib/np.sum(BunchProfiles[:, 0])
    return timeScale_for_tomo, BunchProfiles


def getTimeProfiles_FromData_2(fname, Ib):
    with hp.File(fname, 'r') as sf:
        BunchProfiles = np.array(sf['bunchProfiles'])
        EnergyProfiles = np.array(sf['energyProfiles'])
        phaseSpace_density_array = np.array(sf['phaseSpace_density_array'])
        x_bin_center_array = np.array(sf['x_bin_center_array'])
        y_bin_center_array = np.array(sf['y_bin_center_array'])
    with open(fname, 'rb') as f:
        timeScale_for_tomo = np.load(f)
        BunchProfiles = np.load(f)
    BunchProfiles = BunchProfiles*Ib/np.sum(BunchProfiles[:, 0])
    return timeScale_for_tomo, BunchProfiles


def getTimgForModelFromDataFile(fname, Ib, T_normFactor, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, centroid_offset=0):
    timeScale_for_tomo, BunchProfiles = getTimeProfiles_FromData(fname, Ib)
    BunchProfiles = BunchProfiles/T_normFactor
    sel_turns = np.arange(start_turn, skipturns *
                          (IMG_OUTPUT_SIZE-2*zeropad), skipturns).astype(np.int32)
    T_img = np.pad(BunchProfiles[:, sel_turns], ((zeropad-centroid_offset, zeropad +
                                                  centroid_offset), (zeropad, zeropad)), 'constant', constant_values=(0, 0))
    T_img_ForModel = normalizeIMG(np.reshape(T_img, T_img.shape+(1,)))
    return T_img_ForModel


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


def normalize(vec):
    return vec/np.max(vec)
