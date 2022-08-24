# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 11:01:14 2020

@author: teoarg

Created on Wed Sep  2 16:52:10 2020

@author: gtrad

"""
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import h5py as hp
import pickle as pk
import random
import time
from IPython import display
import shutil

from utils import extract_data_Fromfolder,load_model_data
from models import extendedCED, compute_mse_physical_loss

IMG_OUTPUT_SIZE = 128
zeropad = 14
start_turn = 1  # skip first turn from theo simulations
skipturns = 3
BUFFER_SIZE = 50
BATCH_SIZE = 32 #8

#%% GENERATE ML DATA
# simulations_dir = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\results_tomo_test'
# save_dir = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\Turn_pickles_Arr_test'
simulations_dir = '/eos/user/k/kiliakis/tomo_data/results_tomo'
save_dir = '/eos/user/k/kiliakis/tomo_data/datasets'

#print(os.listdir(simulations_dir))
readSimData =True
generateMLData = True
if generateMLData:
    if readSimData:
        print('Generating ML data from Theo simulations, importing it from ', simulations_dir)  
        SIM_DATA = []
        maxE, maxT, maxB = [], [], []
        for i,fn in enumerate(os.listdir(simulations_dir)):
            #print(i, fn)
            try:
                paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
                    extract_data_Fromfolder(fn, simulations_dir, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns)
            except:
                print('NOT VALID', fn)
            SIM_DATA.append({'fn':fn,
                             'params':paramsDict,
                             'skipturns':skipturns,
                             'turns':sel_turns,
                             'E_img':E_img,
                             'T_img':T_img,
                             'B_img':PS_img_dec})
            maxE.append(np.max(E_img))
            maxT.append(np.max(T_img))
            maxB.append(np.max(PS_img_dec.flatten()))
    
        norm_SIM_DATA = []
        E_normFactor =  np.max(maxE)
        T_normFactor =  np.max(maxT)
        B_normFactor =  np.max(maxB)
        
        for simDict in SIM_DATA:
            simDict['E_img'] = simDict['E_img']/E_normFactor
            simDict['T_img'] = simDict['T_img']/T_normFactor
            for i in range(np.shape(simDict['B_img'])[2]):
                simDict['B_img'][:,:,i] = simDict['B_img'][:,:,i]/B_normFactor
            simDict['params']['E_normFactor']=E_normFactor
            simDict['params']['T_normFactor']=T_normFactor
            simDict['params']['B_normFactor']=B_normFactor
            norm_SIM_DATA.append(simDict)
        
        print('Saving ALL data in single pickle')
        pk.dump( {'norm_SIM_DATA':norm_SIM_DATA}, open( os.path.join(save_dir,"ALL_DATA_normalized.pk"), "wb" ))
        
    else:
        print('Loading all data normalized')
        # data_Dict = pk.load(open(os.path.join(save_dir,"ALL_DATA_normalized.pk"),"rb"))
        locals().update(pk.load(open(os.path.join(save_dir,"ALL_DATA_normalized.pk"),"rb")))
    

    random.shuffle(norm_SIM_DATA)
    random.shuffle(norm_SIM_DATA)
    random.shuffle(norm_SIM_DATA)
    
    ML_dir = os.path.join(save_dir,'ML_data')
    ML_dir_directTPS = os.path.join(ML_dir, 'directTPS')
    ML_dir_directTPS_test = os.path.join(ML_dir, 'directTPS_test')
    os.mkdir(ML_dir)
    os.mkdir(ML_dir_directTPS)
    os.mkdir(ML_dir_directTPS_test)
    
    num_Cases = 300 # out of the 1200 cases simulated
    num_Turns_Case = 50 # out of the 100 turns selected by case (1 out of 3, so in max 300 turns)
    num_Cases_test = 50 
    num_Turns_Case_test = 1   
    
    for i, normSimDict in enumerate(norm_SIM_DATA[:num_Cases]):
        print(i)  
        for turn in [0]+random.choices(normSimDict['turns'][1:], k=num_Turns_Case):
            pk.dump( {'turn': turn,
                      'T_img':normSimDict['T_img'],
                      'params':normSimDict['params'],
                      'fn':normSimDict['fn'],
                      'PS': normSimDict['B_img'][:,:,turn//skipturns]}, 
                      # 'PS': normSimDict['B_img'][:,:,turn//normSimDict['skipturns']]}, 
                    open( os.path.join(ML_dir_directTPS,"{}.pk".format(i)), "wb" ) ) 
    
    for i, normSimDict in enumerate(norm_SIM_DATA[num_Cases:num_Cases+num_Cases_test]):
        print(i)  
        for turn in [0]+random.choices(normSimDict['turns'][1:], k=num_Turns_Case_test):
            pk.dump( {'turn': turn,
                      'T_img':normSimDict['T_img'],
                      'params':normSimDict['params'],
                      'fn':normSimDict['fn'],
                      'PS': normSimDict['B_img'][:,:,turn//skipturns]}, 
                      # 'PS': normSimDict['B_img'][:,:,turn//normSimDict['skipturns']]}, 
                    open( os.path.join(ML_dir_directTPS_test,"{}.pk".format(i)), "wb" ) ) 
    
    training_ratio=0.8
    X = os.listdir(ML_dir_directTPS)
    TRAINING_PATH = os.path.join(ML_dir_directTPS, 'TRAINING')
    VALIDATION_PATH = os.path.join(ML_dir_directTPS, 'VALIDATION')
    TESTING_PATH = ML_dir_directTPS_test
    os.mkdir(TRAINING_PATH)
    os.mkdir(VALIDATION_PATH)
    
    for i, fn in enumerate(X):
        if i<=len(X)*training_ratio:
            shutil.copy(os.path.join(ML_dir_directTPS,fn), os.path.join(TRAINING_PATH,fn))
        else:
            shutil.copy(os.path.join(ML_dir_directTPS,fn), os.path.join(VALIDATION_PATH,fn))

#%% Folders

weights_dir  = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\Turn_pickles_Arr_3\\weights'
TESTING_PATH = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\Turn_pickles_Arr_3\\TESTING_DATA'
# TESTING_PATH = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\Turn_pickles_Arr_2\\TESTING_DATA'
TRAINING_PATH = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\Turn_pickles_Arr_3\\TRAINING_DATA\\ML_data\\directTPS'
simulations_dir = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\results_tomo_3'
dataRun2_dir= 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'

simulations_dir_2 = 'G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\results_tomo_test'


#%%
generateMLData= False
readSimData = False


#%% CREATE ML DATASETS




training_dataset = tf.data.Dataset.list_files(TRAINING_PATH + '/*.pk')
training_dataset = training_dataset.map(lambda x: tf.py_function(load_model_data, [x], [tf.float32, tf.float32, tf.float32, tf.string,\
                                                                                tf.float32, tf.float32, tf.float32, tf.float32,tf.float32, tf.float32,\
                                                                                tf.float32, tf.float32]))

training_dataset = training_dataset.shuffle(BUFFER_SIZE)
training_dataset = training_dataset.batch(BATCH_SIZE)
    
test_dataset = tf.data.Dataset.list_files(TESTING_PATH + '/*.pk')
test_dataset = test_dataset.map(lambda x: tf.py_function(load_model_data, [x], [tf.float32, tf.float32, tf.float32, tf.string,\
                                                                                tf.float32, tf.float32, tf.float32, tf.float32,tf.float32, tf.float32,\
                                                                                tf.float32, tf.float32]))    
# test_dataset = test_dataset.map(lambda x: tf.py_function(load_model_data_old, [x], [tf.float32, tf.float32, tf.float32, tf.string,\
#                                                                                 tf.float32, tf.float32, tf.float32, tf.float32,\
#                                                                                 tf.float32, tf.float32]))    
test_dataset = test_dataset.shuffle(BUFFER_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)


#%%
# Model instantiation 

input_shape=(IMG_OUTPUT_SIZE,IMG_OUTPUT_SIZE,1)
latent_dim = 6
additional_latent_dim = 1


#eCED = extendedCED(latent_dim, additional_latent_dim, input_shape, filters = [64,128,256,512]) 
eCED = extendedCED(latent_dim, additional_latent_dim, input_shape, filters = [32, 64, 128,256,512, 1024]) # test performance

#%%
# Data to run the model once
for training_normTurns, training_Timgs, training_PSimgs,\
    training_fns,\
    trainingt_phErs, training_enErs, training_bls, training_intens,training_Vrfs, training_mus,\
    training_T_normFactors, training_B_normFactors in training_dataset.take(2):
        i=0
        training_normTurn = training_normTurns[i:i+1]
        training_Timg = training_Timgs[i:i+1]
        training_PSimg = training_PSimgs[i:i+1]    

training_T_normFactor = training_T_normFactors[0].numpy()

for test_normTurns, test_Timgs, test_PSimgs,\
    test_fns,\
    test_phErs, test_enErs, test_bls, test_intens,test_Vrfs, test_mus,\
    _, __ in test_dataset.take(5):
        i=0
        test_normTurn = test_normTurns[i:i+1]
        test_Timg = test_Timgs[i:i+1]
        test_PSimg = test_PSimgs[i:i+1]    

### test with old simulations of constant voltage and mu
# for test_normTurns, test_Timgs, test_PSimgs,\
#     test_fns,\
#     test_phErs, test_enErs, test_bls, test_intens,\
#     _, __ in test_dataset.take(1):
#     i=0
#     test_normTurn = test_normTurns[i:i+1]
#     test_Timg = test_Timgs[i:i+1]
#     test_PSimg = test_PSimgs[i:i+1]    

assess_model(eCED, test_normTurn, test_Timg, test_PSimg, 0)    
assess_model(eCED, training_normTurn, training_Timg, training_PSimg, 0)    
#%%
# Loading weights in the model
eCED.encoder.load_weights(os.path.join(weights_dir, 'eCED_weights_encoder_v3_BI_22Sept_TESTBigger.h5'))
eCED.extender.load_weights(os.path.join(weights_dir, 'eCED_weights_extender_v3_BI_22Sept_TESTBigger.h5'))
eCED.decoder.load_weights(os.path.join(weights_dir, 'eCED_weights_decoder_v3_BI_22Sept_TESTBigger.h5'))

# Retry applying the model after new weights were loaded
assess_model(eCED, test_normTurn, test_Timg, test_PSimg, 0)
assess_model(eCED, training_normTurn, training_Timg, training_PSimg, 0)    




#%%
Expected_params, Predicted_params = [], []
#for n,\
#    (test_normTurns, test_Timgs, test_PSimgs,\
#    test_fns,\
#    test_phErs, test_enErs, test_bls, test_intens, test_Vrfs, test_mus,\
#    test_T_normFactors, test_B_normFactors) in test_dataset.enumerate():
count = 0
# for n, (test_normTurns, test_Timgs, test_PSimgs,test_fns,test_phErs, test_enErs, test_bls, test_intens,test_T_normFactors, test_B_normFactors) in test_dataset.enumerate():    
for n,\
    (test_normTurns, test_Timgs, test_PSimgs,\
    test_fns,\
    test_phErs, test_enErs, test_bls, test_intens, test_Vrfs, test_mus,\
    test_T_normFactors, test_B_normFactors) in test_dataset.enumerate(): 
    print(test_fns)
    for i in range(len(test_fns)):
        count+=1
        print(count)
        test_normTurn = test_normTurns[i:i+1]
        test_Timg = test_Timgs[i:i+1]
        test_PSimg = test_PSimgs[i]
        test_fn = test_fns[i]
        test_phEr = test_phErs[i].numpy()
        test_enEr = test_enErs[i].numpy() 
        test_bl = test_bls[i].numpy() 
        test_inten = test_intens[i].numpy()
        test_Vrf = test_Vrfs[i].numpy()
        test_mu = test_mus[i].numpy()
        test_T_normFactor = test_T_normFactors[i].numpy()
        test_B_normFactor = test_B_normFactors[i].numpy()
        preds, parss = eCED.predictPS(test_Timg*1, test_normTurn)
        pred_phEr, pred_enEr, pred_bl, pred_inten, pred_Vrf, pred_mu = unnormalize_params(*list(parss[0].numpy()))
        Expected_params.append([test_phEr, test_enEr, test_bl, test_inten, test_Vrf, test_mu])
        Predicted_params.append([pred_phEr, pred_enEr, pred_bl, pred_inten, pred_Vrf, pred_mu])
Expected_params = np.array(Expected_params)
Predicted_params = np.array(Predicted_params)

DelParam = Predicted_params[:,0:6]-Expected_params
DelParam.shape
plt.figure();plt.hist(DelParam[:,0],bins=100,range=(-50,50));plt.xlabel('Phase Error Diff [deg]')
plt.figure();plt.hist(DelParam[:,1],bins=100,range=(-100,100));plt.xlabel('Energy Error Diff [MeV]')
plt.figure();plt.hist(DelParam[:,2]*1e12,bins=100,range=(-500,500));plt.xlabel('length Diff [ps]')
plt.figure();plt.hist(DelParam[:,3]*1e-10,bins=100,range=(-10.5,10.5));plt.xlabel('I diff [1e9 prot]')
plt.figure();plt.hist(DelParam[:,4],bins=100,range=(-0.3,0.3));plt.xlabel('Vrf diff [MV]')
plt.figure();plt.hist(DelParam[:,5],bins=100,range=(-0.5,0.5));plt.xlabel('mu diff [a.u.]')
plt.figure();plt.plot(Predicted_params[:,4])

#%% Until here checked
from generalFunctions import ring_params,fs0_calculation,calculateEmittance  
#

y_array = np.linspace(-500,500,100)
x_array = np.linspace(0,2.5,100)


# plt.rcParams['figure.figsize'] = [10, 7]
for test_normTurns, test_Timgs, test_PSimgs,\
    test_fns,\
    test_phErs, test_enErs, test_bls, test_intens, test_Vrfs, test_mus,\
    test_T_normFactors, test_B_normFactors in test_dataset.take(2):
    i = np.random.randint(len(test_fns))
    test_normTurn = test_normTurns[i:i+1]
    test_Timg = test_Timgs[i:i+1]
    test_PSimg = test_PSimgs[i]
    test_fn = test_fns[i]
    test_phEr = test_phErs[i].numpy()
    test_enEr = test_enErs[i].numpy() 
    test_bl = test_bls[i].numpy() 
    test_inten = test_intens[i].numpy()
    test_Vrf = test_Vrfs[i].numpy()
    test_mu = test_mus[i].numpy()
    test_T_normFactor = test_T_normFactors[i].numpy()
    test_B_normFactor = test_B_normFactors[i].numpy()

# for test_normTurns, test_Timgs, test_PSimgs,\
#     test_fns,\
#     test_phErs, test_enErs, test_bls, test_intens,\
#     test_T_normFactors, test_B_normFactors in test_dataset.take(1):
#     i = np.random.randint(len(test_fns))
#     test_normTurn = test_normTurns[i:i+1]
#     test_Timg = test_Timgs[i:i+1]
#     test_PSimg = test_PSimgs[i]
#     test_fn = test_fns[i]
#     test_phEr = test_phErs[i].numpy()
#     test_enEr = test_enErs[i].numpy() 
#     test_bl = test_bls[i].numpy() 
#     test_inten = test_intens[i].numpy()
#     test_Vrf = test_Vrfs[i].numpy()
#     test_mu = test_mus[i].numpy()
#     test_T_normFactor = test_T_normFactors[i].numpy()
#     test_B_normFactor = test_B_normFactors[i].numpy()


rf_voltage_main = test_Vrf # MV
mom = 450 # GeV/c
bucketArea, bunchArea,xbucket,Ubucket,x_bunch,U_bunch,x_phaseSpace,E_phaseSpace,x_bunchPhaseSpace,E_bunchPhaseSpace = calculateEmittance(rf_voltage_main,1,0.,mom*1e3,bunchLength=1.2e-9)  
#x_phaseSpace*=180/np.pi
x_phaseSpace*=1e9/(2*np.pi*400e6)
x_bunchPhaseSpace*=1e9/(2*np.pi*400e6)




# # create  a turn by turn tomo prediction from T profile
# paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
#                     extract_data_Fromfolder(test_fn.numpy().decode(), simulations_dir, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)



# create  a turn by turn tomo prediction from T profile in simulations
filename = 'phEr0.0_enEr0_bl1.36e-09_int1.18e+11_Vrf2.00_mu1.5'
filename = 'phEr0.0_enEr0_bl1.36e-09_int1.19e+11_Vrf3.90_mu1.5'

data_path = r'\\afs\cern.ch\\work\\t\\teoarg\\python\\LHC\\protons\\tomography\\local_sim\\sim_results'
paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
                    extract_data_Fromfolder(filename, data_path, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)


# # create  a turn by turn tomo prediction from T profile
# paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
#                     extract_data_Fromfolder('phEr9_enEr-67_bl1.40e-09_int1.14e+11_Vrf4.0_mu2.1', simulations_dir_2, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)


# # create  a turn by turn tomo prediction from T profile
# _, _, _, _, T_img_old_imp, _ = \
#                     extract_data_Fromfolder('phEr9_enEr-67_bl1.30e-09_int1.43e+11_Vrf4.0_mu2.1', simulations_dir, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)

############
# T_img_temp = T_img/ T_normFactor
# T_img_for_model = normalizeIMG(np.reshape(T_img_temp, T_img_temp.shape+(1,)))
# T_img_for_model = np.reshape(normalizeIMG(T_img/ T_normFactor),(128, 128,1))
T_img_for_model = np.reshape(normalizeIMG(T_img/ test_T_normFactor),(128, 128,1))


#####################

# for turn in np.arange(1,300,300):
#   plt.figure()
#   plt.imshow(PS_imgs[:,:,turn],  vmin=0, vmax =test_B_normFactor,extent=[x_array[0],x_array[-1],y_array[0],y_array[-1]], origin = 'lower',aspect='auto',);
#   plt.title('Turn - {}'.format(turn))
#   # plt.gca().get_xaxis().set_visible(False)
#   # plt.gca().get_yaxis().set_visible(False)

factor = 1 #0.56258553#test_T_normFactor/training_T_normFactor
#factor = training_T_normFactor/test_T_normFactor
for turn in np.arange(0,300,20):
    
    t0 = time.perf_counter()
    norm_turn = normalizeTurn(turn)
    f,ax = plt.subplots(2,2)
    # preds, parss= eCED.predictPS(test_Timg*factor, tf.expand_dims(norm_turn,axis =0))
    # preds, parss= eCED.predictPS(T_imag_2*factor, tf.expand_dims(norm_turn,axis =0))
    
    
    preds, parss = eCED.predictPS(tf.expand_dims(T_img_for_model, axis=0),
                                        tf.expand_dims(norm_turn, axis=0))
    
    extracted_parameters = unnormalize_params(*list(parss[0].numpy()))
    # print(test_phEr,test_enEr,test_bl,test_inten,test_Vrf,test_mu)
    print(extracted_parameters)
    tmp_predicted_PS = unnormalizeIMG(preds[0,:,:,0])*test_B_normFactor
    tmp_expected_PS = PS_imgs[:,:,turn]
    
    # ax[0,0].imshow(tmp_predicted_PS, vmin=0, vmax =test_B_normFactor, aspect='auto',extent=[x_array[0],x_array[-1],y_array[0],y_array[-1]], origin = 'lower',cmap=cmap_white_blue_red); ax[0,0].set_title('Predicted\nTurn - {}'.format(turn))
    ax[0,0].imshow(tmp_predicted_PS, vmin=0,  aspect='auto',extent=[x_array[0],x_array[-1],y_array[0],y_array[-1]], origin = 'lower',cmap=cmap_white_blue_red); ax[0,0].set_title('Predicted\nTurn - {}'.format(turn))
    ax[0,0].plot(x_phaseSpace,E_phaseSpace*1e-6,color='r')
    ax[0,0].plot(x_bunchPhaseSpace,E_bunchPhaseSpace*1e-6,color='k')
    # ax[0,1].imshow(tmp_expected_PS, vmin=0, vmax =test_B_normFactor, aspect='auto',extent=[x_array[0],x_array[-1],y_array[0],y_array[-1]], origin = 'lower',cmap=cmap_white_blue_red); ax[0,1].set_title('Expected\nTurn - {}'.format(turn))
    ax[0,1].imshow(tmp_expected_PS, vmin=0, aspect='auto',extent=[x_array[0],x_array[-1],y_array[0],y_array[-1]], origin = 'lower',cmap=cmap_white_blue_red); ax[0,1].set_title('Expected\nTurn - {}'.format(turn))
    ax[0,1].plot(x_phaseSpace,E_phaseSpace*1e-6,color='r')
    ax[0,1].plot(x_bunchPhaseSpace,E_bunchPhaseSpace*1e-6,color='k')
    ax[1,0].plot(x_array,np.sum(tmp_predicted_PS,0)[14:114], label = 'Predicted')
    ax[1,0].plot(x_array,np.sum(tmp_expected_PS,0)[14:114], label = 'Expected')
    ax[1,0].legend(); ax[1,0].set_title('Time')
    ax[1,1].plot(y_array,np.sum(tmp_predicted_PS,1)[14:114], label = 'Predicted')
    ax[1,1].plot(y_array,np.sum(tmp_expected_PS,1)[14:114], label = 'Expected')
    ax[1,1].legend(); ax[1,1].set_title('Energy')
    
    print(time.perf_counter()-t0)


# predicted_T_profiles={}                    
# for turn in np.arange(1,300):
#   norm_turn = normalizeTurn(turn)
#   preds, _ = eCED.predictPS(test_Timg, tf.expand_dims(norm_turn,axis =0))
#   tmp_predicted_PS = unnormalizeIMG(preds[0,:,:,0])*test_B_normFactor
#   tmp_expected_PS = PS_imgs[:,:,turn]
#   predicted_T_profiles[turn]=np.sum(tmp_predicted_PS,0)


# plt.imshow(test_Timg[0,:,:,0]);plt.colorbar()
# plt.imshow(unnormalizeIMG(test_Timg[0,:,:,0]));plt.colorbar()
# plt.imshow(unnormalizeIMG(test_Timg[0,:,:,0])*test_T_normFactor); plt.colorbar()
# plt.plot(np.sum(unnormalizeIMG(test_Timg[0,:,:,0])*test_T_normFactor,0));plt.ylim((1.28e11,1.32e11));plt.xlim((14,114))
# plt.imshow(tmp_expected_PS);plt.colorbar()

## plot all profiles
#real_testTProfs = unnormalizeIMG(test_Timg[0,:,:,0])*test_T_normFactor
#for i in range(zeropad, real_testTProfs.shape[1]-zeropad):
#    turn = (i-zeropad)*skipturns+1
#    plt.figure()
#    plt.plot(real_testTProfs[:,i])
#    plt.plot(predicted_T_profiles[turn])



#%%
from fitFunctions import fwhm

# create  a turn by turn tomo prediction from T profile in simulations
filename = 'phEr0.0_enEr0_bl1.36e-09_int1.18e+11_Vrf2.00_mu1.5'
filename = 'phEr0.0_enEr0_bl1.36e-09_int1.19e+11_Vrf3.90_mu1.5'

data_path = r'\\afs\cern.ch\\work\\t\\teoarg\\python\\LHC\\protons\\tomography\\local_sim\\sim_results'
paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
                    extract_data_Fromfolder(filename, data_path, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)
                    
# # create  a turn by turn tomo prediction from T profile
# paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
#                     extract_data_Fromfolder('phEr9_enEr-67_bl1.40e-09_int1.14e+11_Vrf4.0_mu2.1', simulations_dir_2, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)
# # paramsDict, PS_imgs, sel_turns, E_img, T_img, PS_img_dec = \
# #                     extract_data_Fromfolder('phEr9_enEr-67_bl1.30e-09_int1.43e+11_Vrf4.0_mu2.1', simulations_dir_2, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)                    

# create  a turn by turn tomo prediction from T profile
_, _, _, _, T_img_old_imp,PS_img_old_imp = \
                    extract_data_Fromfolder('phEr9_enEr-67_bl1.30e-09_int1.43e+11_Vrf4.0_mu2.1', simulations_dir, IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns, version=3)


                    
                    
simulated_T_profiles={}  
simulated_T_profiles_old_imp={}                  
for turn in np.arange(1,100):
  simulated_T_profiles[turn]=np.sum(PS_img_dec[:,:,turn],0)
  simulated_T_profiles_old_imp[turn]=np.sum(PS_img_old_imp[:,:,turn],0)                    


ML_BPs = np.zeros((100,len(simulated_T_profiles)))
ML_BPs_old_imp = np.zeros((100,len(simulated_T_profiles_old_imp)))
for i in np.arange(1,len(simulated_T_profiles)+1):
#    print(predicted_T_profiles[i])
    ML_BPs[:,i-1] = simulated_T_profiles[i][14:114]
    ML_BPs_old_imp[:,i-1] = simulated_T_profiles_old_imp[i][14:114]

Trev = 88.9e-6
timeScale_ML_BPs = np.arange(0,np.shape(ML_BPs)[0])*25e-12 #in ns
# ML_BPs *= 1/np.max(ML_BPs)*2 

b_position = np.array([])
bl = np.array([])
b_position_old_imp = np.array([])
bl_old_imp = np.array([])

for i in np.arange(np.shape(ML_BPs)[1]):   
    y = ML_BPs[:,i]
    (mu, sigma, amp) = fwhm(timeScale_ML_BPs,y,level=0.5)
    b_position = np.append(b_position,mu)
    bl = np.append(bl,4*sigma)
    
    y = ML_BPs_old_imp[:,i]
    (mu, sigma, amp) = fwhm(timeScale_ML_BPs,y,level=0.5)
    b_position_old_imp = np.append(b_position_old_imp,mu)
    bl_old_imp = np.append(bl_old_imp,4*sigma)
    

# b_position_data = running_mean(b_position_data, 1)
# plt.figure();plt.plot(bl)
# plt.plot(bl_old_imp) 
plt.figure();plt.plot((b_position-np.mean(b_position))*360/2.5e-9)
plt.plot((b_position_old_imp-np.mean(b_position_old_imp))*360/2.5e-9)
plt.xlabel('# Turns',fontsize=14)
plt.ylabel('Bunch position [ps]',fontsize=14)



plt.figure();plt.plot(bl)
plt.plot(bl_old_imp) 
plt.xlabel('# Turns',fontsize=14)
plt.ylabel('Bunch length [s]',fontsize=14)
        
#%% TRZ TO SIMULATE

# simPS = eCED.decode(eCED.extend(tf.expand_dims(normalize_params(0, 0, 1.5e-9, 111, 4, 2), axis=0),
#                                 tf.expand_dims(normalizeTurn(0), axis=0)))

# plt.figure();plt.imshow(simPS[0,:,:,0])

plt.figure()
for mu in np.linspace(0.5,5,10):
    simPS = eCED.decode(eCED.extend(tf.expand_dims(normalize_params(6, 0, 1.5e-9, 1.16e11, 4,mu), axis=0),
                                tf.expand_dims(normalizeTurn(0), axis=0)))
    plt.plot(np.sum(unnormalizeIMG(simPS[0,:,:,0])*B_normFactor,0))
plt.plot(np.arange(14,114),real_BunchProfiles[:,0],'k')



plt.figure()
for inten in np.linspace(0.8,1.6,10)*1e11:
    simPS = eCED.decode(eCED.extend(tf.expand_dims(normalize_params(6, 0, 1.5e-9, inten, 4,2), axis=0),
                                tf.expand_dims(normalizeTurn(200), axis=0)))
    proj = np.sum(unnormalizeIMG(simPS[0,:,:,0])*B_normFactor,0)
    plt.plot(proj/proj[64])
plt.plot(np.arange(14,114),real_BunchProfiles[:,0],'k')



#%% On data
X=[]
fileName = 'PROFILE_B2_b30660_20180908022914.npy'# int 1.16e11
#fileName = 'PROFILE_B2_b30850_20180908022914.npy' # int 1.132e11
#fileName = 'PROFILE_B2_b31130_20180908022914.npy' # int 1.03e11
#fileName = 'PROFILE_B2_b19830_20180930122142.npy' # int 1.105e11
#fileName = 'PROFILE_B2_b20290_20180930122142.npy' # int 1.201e11

# fileName = 'PROFILE_B1_b25540_20180616020329.npy' # at injection 6 MV 10 degrees phase error int 1.14e11
# fileName = 'PROFILE_B1_b25540_20180616043102.npy' # at injection 4 MV 10 degrees phase error int 1.02773e11

#fileName = 'PROFILE_B1_b1_20180915194817.npy' # at injection 4 MV,int 1.137e11 single bunch
#fileName = 'PROFILE_B1_b2001_20180915195046.npy' # at injection 4 MV,int 1.216e11 single bunch 
#fileName = 'PROFILE_B1_b3001_20180915195315.npy' # at injection 4 MV,int 1.1173e11 single bunch 
#fileName = 'PROFILE_B2_b19001_20180915201534.npy' # at injection 4 MV,int 1.160e11 single bunch 

intensity = 1.16e11
#intensity -= 0.35*intensity
for centroid_offset in [0]: #[2]: #np.arange(0,10):
    dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'
    TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, fileName), intensity,
                                                    test_T_normFactor,
                                                    IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                    centroid_offset=centroid_offset)
    turn_id = 0.0
    predicted_PS, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                          tf.expand_dims(normalizeTurn(turn_id), axis=0))
    extracted_parameters = unnormalize_params(*list(parss[0].numpy()))
    print(extracted_parameters)


    predicted_T_profiles={}
    # Predict evolution for 300 turns
    for turn_id in np.arange (0, 300):
      predicted_PS, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0), tf.expand_dims(normalizeTurn(turn_id), axis=0))
      predicted_PS=unnormalizeIMG(predicted_PS[0,:,:,0])*test_B_normFactor
      predicted_T_profiles[turn_id] = np.sum(predicted_PS,0)
    
    
    # compare time profiles with your tomo
    timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, fileName), intensity)
    
    tmp=[]
    for turn_id in np.arange(0,300, 1):
      tmp.append(np.mean((real_BunchProfiles[:,turn_id]-predicted_T_profiles[turn_id][14:114])**2))
    tmp = np.array(tmp)
    X.append([centroid_offset, np.sum(tmp)])
X = np.array(X)

#plt.figure()
#plt.plot(X[:,0], X[:,1])


##%%


ML_BPs = np.zeros((100,len(predicted_T_profiles)))
for i in np.arange(len(predicted_T_profiles)):
#    print(predicted_T_profiles[i])
    ML_BPs[:,i] = predicted_T_profiles[i][14:114]
    

Trev = 88.9e-6
timeScale_ML_BPs = np.arange(0,np.shape(ML_BPs)[0])*25e-12 #in ns
ML_BPs *= 1/np.max(ML_BPs)*2 

b_position = np.array([])
bl = np.array([])
b_position_data = np.array([])
bl_data = np.array([])

for i in np.arange(np.shape(ML_BPs)[1]):   
    y = ML_BPs[:,i]
    (mu, sigma, amp) = fwhm(timeScale_ML_BPs,y,level=0.5)
    b_position = np.append(b_position,mu)
    bl = np.append(bl,4*sigma)
    
    y = real_BunchProfiles[:,i]
    (mu, sigma, amp) = fwhm(timeScale_ML_BPs,y,level=0.5)
    b_position_data = np.append(b_position_data,mu)
    bl_data = np.append(bl_data,4*sigma)
    

b_position_data = running_mean(b_position_data, 1)
plt.figure();plt.plot(bl)
plt.plot(bl_data) 
plt.figure();plt.plot((b_position-np.mean(b_position))*360/2.5e-9)
plt.plot((b_position_data-np.mean(b_position_data))*360/2.5e-9)
plt.xlabel('# Turns',fontsize=14)
plt.ylabel('Bunch position [ps]',fontsize=14)



plt.figure();plt.plot(bl[::3])
plt.plot(bl_data) 
plt.xlabel('# Turns',fontsize=14)
plt.ylabel('Bunch length [s]',fontsize=14)
#%%



dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'
fname = os.path.join(dataRun2_dir, fileName)
with open(fname, 'rb') as f:
    timeScale_for_tomo = np.load(f)
    BunchProfiles = np.load(f) 
    for i in np.arange(BunchProfiles.shape[1]):
        BunchProfiles[:,i] = symmetrizeProfile(timeScale_for_tomo,BunchProfiles[:,i])

    
y = real_BunchProfiles[:,0]
x = timeScale_ML_BPs
y = symmetrizeProfile(x,y)
(mu, sigma, amp) = fwhm(timeScale_ML_BPs,y,level=0.5)

y_qg = qgaussian_pdf(1e9*timeScale_ML_BPs,mu=1e9*mu,amp=amp,q=0.55,beta=3.2)
    
tau_2 = 2.*sigma
yy = binomial_pdf(timeScale_ML_BPs, mu=mu, amp=amp, m=1.9, tau_2=tau_2*1.2)

# yyyy = (1-((timeScale_ML_BPs-mu)/tau_2)**2)
# yyyy[yyyy<0] =0 
plt.figure()
# plt.plot(timeScale_ML_BPs,yyyy**2)
plt.plot(timeScale_ML_BPs,y)
plt.plot(timeScale_ML_BPs,yy)
plt.plot(timeScale_ML_BPs,y_qg)
#%% After the scan of centroid parametr:
from generalFunctions import ring_params,fs0_calculation,calculateEmittance  
#
rf_voltage_main = 4 # MV
mom = 450 # GeV/c
bucketArea, bunchArea,xbucket,Ubucket,x_bunch,U_bunch,x_phaseSpace,E_phaseSpace,x_bunchPhaseSpace,E_bunchPhaseSpace = calculateEmittance(rf_voltage_main,1,0.,mom*1e3)  
x_phaseSpace*=180/np.pi
    
#dtBin = np.double(linecache.getline(plotInfo, 6)[9:-1])
#dEBin = np.double(linecache.getline(plotInfo, 8)[9:-1])
y_array = np.linspace(-500,500,100)
x_array = np.linspace(0,360,100)
#    x0 = np.double(linecache.getline(plotInfo, 12)[8:-1])
#    y0 = np.double(linecache.getline(plotInfo, 13)[8:-1])
#    
#    x_array = np.arange(0, profLen*dtBin, dtBin)[0:profLen] - x0*dtBin
#    y_array = np.arange(0, profLen*dEBin, dEBin)[0:profLen] - y0*dEBin#-0.175e8


    
plt.rcParams['figure.figsize'] = [7, 7]
centroid_offset = 0
dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'

#fileName = 'PROFILE_B2_b30670_20180908022914.npy'
fileName = 'PROFILE_B2_b30660_20180908022914.npy' # int 1.16e11
#fileName = 'PROFILE_B2_b30850_20180908022914.npy' # int 1.132e11
#fileName = 'PROFILE_B2_b31130_20180908022914.npy' # int 1.03e11
# fileName = 'PROFILE_B1_b25540_20180616043102.npy' # int 1.03e11
#fileName = 'PROFILE_B2_b19830_20180930122142.npy' # int 1.105e11
#fileName = 'PROFILE_B2_b20300_20180930122142.npy' # int 1.081e11
#fileName = 'PROFILE_B2_b20290_20180930122142.npy' # int 1.201e11

# fileName = 'PROFILE_B1_b25540_20180616020329.npy' # at injection 6 MV 10 degrees phase error int 1.14e11


# fileName = 'PROFILE_B1_b1_20180915194817.npy' # at injection 4 MV,int 1.137e11 single bunch
#fileName = 'PROFILE_B1_b2001_20180915195046.npy' # at injection 4 MV,int 1.1216e11 single bunch 
#fileName = 'PROFILE_B1_b3001_20180915195315.npy' # at injection 4 MV,int 1.1173e11 single bunch 
#fileName = 'PROFILE_B2_b19001_20180915201534.npy' # at injection 4 MV,int 1.160e11 single bunch 


# 
intensity = 1.16e11
# intensity = bbb_intensity[0]
#intensity -= 0.35*intensity

TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, fileName), intensity,
                                                test_T_normFactor,
                                                IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                centroid_offset=centroid_offset)    
timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, fileName),intensity)

for turn_id in np.arange(0,300,20):
#    plt.figure('phase space')
#    plt.clf()
    f,ax = plt.subplots(2,1)
#    f,ax = plt.subplots(2,1,num='phase space')
#    f.canvas.set_window_title('phase space')
    

    #  for a in ax:
    #    a.set_xticklabels([])
    #    a.set_yticklabels([])
    

    preds, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
      
    tmp_predicted_PS = unnormalizeIMG(preds[0,:,:,0])*test_B_normFactor
    tmp_predicted_BP = np.sum(tmp_predicted_PS,0)#-np.mean(np.sum(tmp_predicted_PS,0)[0:6])
#    tmp_predicted_BP = tmp_predicted_BP/np.max(tmp_predicted_BP)
    tmp_predicted_BP = tmp_predicted_BP[14:114]
    tmp_predicted_BP_max_ind = np.argmax(tmp_predicted_BP)
    x_tmp_predicted_BP = np.arange(0,len(tmp_predicted_BP))-tmp_predicted_BP_max_ind
  
    real_BP = real_BunchProfiles[:,turn_id]
#    real_BP = real_BP/np.max(real_BP)
    real_BP_max_ind = np.argmax(real_BP)
    x_real_BP = np.arange(0,len(real_BP))-real_BP_max_ind
  
    #np.arange(14,114),
  
    ax[1].plot(x_tmp_predicted_BP,tmp_predicted_BP, label = 'Predicted')
    ax[1].plot(x_real_BP,real_BP, label = 'Expected')
    ax[1].legend(); #ax[1].set_title('Turn {}'.format(turn_id))
#    ax[1].set_ylim(0,6.0e9)
#    ax[1].set_ylim(0,1.5)
    #ax[1].set_xticklabels([])
    im=ax[0].imshow(tmp_predicted_PS,aspect='auto', vmin=0, vmax =test_B_normFactor, extent=[x_array[0],x_array[-1],y_array[0],y_array[-1]], origin = 'lower',cmap=cmap_white_blue_red,norm=colors.Normalize(0, 0.0001,clip=True)); ax[0].set_title('Predicted - Turn - {}'.format(turn_id))
   
    ax[0].plot(x_phaseSpace,E_phaseSpace*1e-6,color='r')
    ax[0].axvline(x=180,color='k')
    ax[0].axhline(y=0,color='k')
    #plt.ylim((0, 5e9))    .
    plt.subplots_adjust(wspace=0, hspace=0)
          
    extracted_parameters = unnormalize_params(*list(parss[0].numpy()))
    print(extracted_parameters)
  
#    image_folder = r'\\afs\cern.ch\work\t\teoarg\python\LHC\protons\tomography\images_meas'
#    fign =  os.path.realpath(image_folder +'/Turn_{:d}.png'.format(int(turn_id)))
#    plt.savefig(fign,bbox_inches='tight')  
#    f.clf()
#%%



dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'

filter_text = '20180908022914.npy'
#filter_text = '20180616060554.npy'
#filter_text = '20180616043102.npy'
#filter_text = '20180930122142.npy'
#filter_text = '20180616020329.npy'



fNames = filterFileNamesInFolder(data_path=dataRun2_dir,filter_text=filter_text)

#Nps = np.array([1.16,1.13,1.23,1.16,1.14,1.07,1.21,1.14,1.11,1.04,1.12,1.10,1.2,1.14,1.21,1.18,1.15,1.06,1.17,1.13,1.13,1.07,1.14,1.11,1.12,1.09,1.14,1.11,1.09,1.03,
#                      1.08,1.08,1.04,1.02,1.03,1.06,1.10,1.10,1.12,1.13,1.09,1.07,1.11,1.12,1.07,1.07,1.09,1.03])*1e11 #1.5e11

Nps = bbb_intensity[0:48] # got this from saved data for Luis (code is in the bottom)
bunch = np.array([])
ph_error = np.array([])
en_error = np.array([])
bl = np.array([])
Np =  np.array([])
turn_id = 1
for i,fileName in enumerate(fNames):
      
    fileName_parts = str.split(fileName,'_')
    beam = int(fileName_parts[1][1::])
    bunch = np.append(bunch,int(fileName_parts[2][1::]))
    
    centroid_offset = 0
    
    
    intensity = Nps[i]
    TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, fileName), intensity,
                                                    test_T_normFactor,
                                                    IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                    centroid_offset=centroid_offset)    
#    timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, fileName),intensity)
    
    preds, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
    
    extracted_parameters = unnormalize_params(*list(parss[0].numpy()))
    ph_error = np.append(ph_error,extracted_parameters[0])
    en_error = np.append(en_error,extracted_parameters[1])
    bl = np.append(bl,extracted_parameters[2])
    Np =  np.append(Np,extracted_parameters[3])
    
#%%    
plt.figure()
plt.plot(1e12*(ph_error-np.mean(ph_error))*2.5e-9/360,'.-',markersize=14)
plt.axhline(0,color='k')
plt.xlabel('# Bunches',fontsize=15)
plt.ylabel('Normalizes Bunch by bunch position along the batch [ps]',fontsize=15) 

plt.figure()
plt.plot((ph_error-np.mean(ph_error)),'.-',markersize=14)
plt.axhline(0,color='k')
plt.xlabel('# Bunches',fontsize=15)
plt.ylabel('Normalizes Bunch by bunch position along the batch [degrees]',fontsize=15)


plt.figure()
plt.plot(ph_error,'.-',markersize=14)
plt.axhline(0,color='k')
plt.xlabel('# Bunches',fontsize=15)
plt.ylabel('Bunch by bunch position along the batch [degrees]',fontsize=15)
plt.title('Mean Phase error ={:3.2f} degrees'.format(np.mean(ph_error)))


plt.figure()
plt.plot(en_error,'.-',markersize=14)
#plt.axhline(0,color='k')    
plt.xlabel('# Bunches',fontsize=15)
plt.ylabel('Energy error [MeV]',fontsize=15)
plt.title('Mean Energy error ={:3.2f} MeV'.format(np.mean(en_error)))


#%% Get the intensity from saved files where measurements saved from timber in other script LHC_bunchProfiles_analysis.py
save_dir = r'C:\Users\teoarg\cernbox\forLuis'
# fileName = 'PROFILE_B2_b30660_20180908022914.h5' # at injection 4 MV
#fileName = 'PROFILE_B1_b25540_20180616060554.h5' # at injection 4 MV 0 degrees phase error
fileName = 'PROFILE_B1_b25540_20180616043102.h5'  #at injection 4 MV 10 degrees phase error
# fileName = 'PROFILE_B1_b25540_20180616020329.h5' # at injection 6 MV 10 degrees phase error
#fileName = 'PROFILE_B2_b19830_20180930122142.h5'

fileName_parts = str.split(fileName,'_')
bucketNumber = int(fileName_parts[2][1::])

fileNameSave = fileName[0:12]+str(bucketNumber)+'_'+fileName_parts[-1][:-3]+'_BunchPositionVariations.npy'

with open(os.path.join(save_dir,fileNameSave), 'rb') as f:
    bbb_positions = np.load(f)
    bbb_intensity = np.load(f)   
    
plt.figure()
plt.plot(bbb_positions,'.-',markersize=14)

plt.figure()
plt.plot(bbb_intensity,'.-',markersize=14)

#%%

# ls REAL_DATA_Run2/

# **LET's try with data**

# load massaged data
dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'

centroid_offset=0
TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, 'PROFILE_B2_b30660_20180908022914.npy'), 1.16e11,
                                                test_T_normFactor,
                                                IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                centroid_offset=centroid_offset)
plt.figure();plt.imshow(TimgData_ForModel[:,:,0],cmap=cmap_white_blue_red); plt.colorbar()
#predict first turn
turn_id = 0.0
predicted_PS, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
extracted_parameters = unnormalize_params(*list(parss[0].numpy()))
predicted_PS = unnormalizeIMG(predicted_PS[0,:,:,0])*test_B_normFactor

plt.figure();
plt.imshow(predicted_PS, origin='lower',cmap=cmap_white_blue_red); plt.colorbar();plt.title('Turn {}'.format(turn_id))
print(extracted_parameters)

#%% Useless for transfer functions

predicted_T_profiles={}
# Predict evolution for 300 turns
for turn_id in np.arange (0, 300):
  predicted_PS, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0), tf.expand_dims(normalizeTurn(turn_id), axis=0))
  predicted_PS=unnormalizeIMG(predicted_PS[0,:,:,0])*test_B_normFactor
  predicted_T_profiles[turn_id] = np.sum(predicted_PS,0)


# compare time profiles with your tomo
timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, 'PROFILE_B2_b30660_20180908022914.npy'), 1.16e11)

TFs=[]
for turn_id in np.arange(0,300, 1):
    plt.figure()
    plt.plot(real_BunchProfiles[:,turn_id], label ='real')
    plt.plot(predicted_T_profiles[turn_id][14:114], label ='Predicted')
    # plt.plot(normalize(real_BunchProfiles[:,turn_id]), label ='real')
    # plt.plot(normalize(predicted_T_profiles[turn_id][14:114]), label ='Predicted')
    plt.title('Turn {}'.format(turn_id))
    plt.legend()
    plt.ylim((0,7e9))
    
    ps_real = np.abs(np.fft.fft(real_BunchProfiles[:,turn_id]))**2
    ps_pred = np.abs(np.fft.fft(predicted_T_profiles[turn_id][14+centroid_offset:114+centroid_offset]))**2
    TFs.append(ps_real/ps_pred)


plt.figure()
for i, TF in enumerate(TFs):
    if i <=200:
        plt.semilogy(TF)
TFs = np.array(TFs)
plt.semilogy(np.sum(TFs[0:200,:],0))



#%% After the scan of centroid parametr:
plt.rcParams['figure.figsize'] = [14, 7]
centroid_offset = 0
dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'
TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, 'PROFILE_B2_b30660_20180908022914.npy'), 1.16e11,
                                                test_T_normFactor,
                                                IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                centroid_offset=centroid_offset)    
timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, 'PROFILE_B2_b30660_20180908022914.npy'), 1.16e11)

for turn_id in np.arange(1,300,1):
  f,ax = plt.subplots(1,2)
  preds, _ = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
  
  tmp_predicted_PS = unnormalizeIMG(preds[0,:,:,0])*test_B_normFactor
  
  ax[0].imshow(tmp_predicted_PS, vmin=0, vmax =test_B_normFactor, origin = 'lower'); ax[0].set_title('Predicted\nTurn - {}'.format(turn_id))
  ax[1].plot(np.sum(tmp_predicted_PS,0), label = 'Predicted')
  ax[1].plot(np.arange(14,114),real_BunchProfiles[:,turn_id], label = 'Expected')
  ax[1].legend(); ax[1].set_title('Turn {}'.format(turn_id))
  plt.ylim((0, 5e9))
  

#%%


# ls REAL_DATA_Run2/

# **LET's try with data**

# load massaged data
dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'

centroid_offset=0
TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, 'PROFILE_B1_b25570_20180616020329.npy'), 1.2e11,
                                                test_T_normFactor,
                                                IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                centroid_offset=centroid_offset)
plt.figure();plt.imshow(TimgData_ForModel[:,:,0]); plt.colorbar()
#predict first turn
turn_id = 0.0
predicted_PS, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
extracted_parameters = unnormalize_params(*list(parss[0].numpy()))
predicted_PS = unnormalizeIMG(predicted_PS[0,:,:,0])*test_B_normFactor

plt.figure();
plt.imshow(predicted_PS, origin='lower'); plt.colorbar();plt.title('Turn {}'.format(turn_id))
print(extracted_parameters)




timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, 'PROFILE_B1_b25570_20180616020329.npy'), 1.2e11)

for turn_id in np.arange(1,300,10):
  f,ax = plt.subplots(1,2)
  preds, _ = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
  
  tmp_predicted_PS = unnormalizeIMG(preds[0,:,:,0])*test_B_normFactor
  
  ax[0].imshow(tmp_predicted_PS, vmin=0, vmax =test_B_normFactor, origin = 'lower'); ax[0].set_title('Predicted\nTurn - {}'.format(turn_id))
  ax[1].plot(np.sum(tmp_predicted_PS,0), label = 'Predicted')
  ax[1].plot(np.arange(14,114),real_BunchProfiles[:,turn_id], label = 'Expected')
  ax[1].legend(); ax[1].set_title('Time')
  plt.ylim((0, 5e9))
  
  
#%%

Ib = 1.16e11
real_tomo_dir = r'G:\Departments\BE\Groups\OP\Sections\LHC\George_Theo\real_tomo\output'#os.path.join(save_dir,'real_tomo\\output')
X = os.listdir(real_tomo_dir)
profLen = 100
real_tomo_by_turn={}
for fn in X:
    if 'image' in fn:
        turn_id = int(fn[5:-5])        
        real_tomo_by_turn[turn_id] = np.ascontiguousarray(np.loadtxt(os.path.join(real_tomo_dir,fn))).reshape((profLen,profLen)).T
        

tomoed_turns = list(real_tomo_by_turn.keys())
scalingFactor = (Ib/np.sum(real_tomo_by_turn[1]))

for turn_id in tomoed_turns:
    real_tomo_by_turn[turn_id] = real_tomo_by_turn[turn_id]*scalingFactor    
    
plt.figure()
for turn_id in tomoed_turns:
    plt.plot(turn_id, np.sum(real_tomo_by_turn[turn_id]), 's')
    
    
for turn_id in tomoed_turns:
    plt.figure()
    plt.imshow(real_tomo_by_turn[turn_id])    
    plt.title('Turn {}'.format(turn_id))
    
  
centroid_offset = 0
dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'
TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, 'PROFILE_B2_b30660_20180908022914.npy'), 1.16e11,
                                                test_T_normFactor,
                                                IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                centroid_offset=centroid_offset) 
   
timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, 'PROFILE_B2_b30660_20180908022914.npy'), 1.16e11)


errs_VEC=[]    

for turn_id in tomoed_turns:
    preds, _ = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
    tmp_predicted_PS = unnormalizeIMG(preds[0,:,:,0])*test_B_normFactor    
    
#    f, ax = plt.subplots(1,4)
#    ax[0].imshow(real_tomo_by_turn[turn_id])    
#    ax[0].set_title('TOMO - Turn {}'.format(turn_id))
#    
#    ax[1].imshow(tmp_predicted_PS[14:114,14:114])    
#    ax[1].set_title('ML - Turn {}'.format(turn_id))
#    
#    # ax[2].plot(np.sum(real_tomo_by_turn[turn_id],0), label = 'Tomo')
#    # ax[2].plot(np.sum(tmp_predicted_PS, 0)[14:114], label = 'ML')
#    # ax[2].plot(real_BunchProfiles[:,turn_id], label = 'Measured')
#    
#    ax[2].plot(normalize(np.sum(real_tomo_by_turn[turn_id], 0)), label = 'Tomo')
#    ax[2].plot(normalize(np.sum(tmp_predicted_PS, 0)[14:114]), label = 'ML')
#    ax[2].plot(normalize(real_BunchProfiles[:,turn_id]), label = 'Measured')    
#    ax[2].legend()
#    
#    ax[3].plot(normalize(real_BunchProfiles[:,turn_id]) - normalize(np.sum(real_tomo_by_turn[turn_id],0)), label = 'ERR Tomo')
#    ax[3].plot(normalize(real_BunchProfiles[:,turn_id]) - normalize(np.sum(tmp_predicted_PS, 0)[14:114]), label = 'ERR ML')    
#    ax[3].legend()
    
    errs_VEC.append([np.mean((normalize(real_BunchProfiles[:,turn_id]) - normalize(np.sum(real_tomo_by_turn[turn_id],0)))**2),
                     np.mean((normalize(real_BunchProfiles[:,turn_id]) - normalize(np.sum(tmp_predicted_PS, 0)[14:114]))**2),
                     np.mean((normalize(np.sum(real_tomo_by_turn[turn_id],0)) - normalize(np.sum(tmp_predicted_PS, 0)[14:114]))**2)])

errs_VEC = np.array(errs_VEC)
plt.figure()
plt.plot(errs_VEC[:,0], label ='tomo vs real')
plt.plot(errs_VEC[:,1], label = 'ML vs real')
plt.plot(errs_VEC[:,2], label = 'ML vs tomo')
plt.xlabel('# Turns*3',fontsize=14)
plt.ylabel('Error [arb. units]',fontsize=14)
plt.legend()



#%%
rf_voltage_main = 4 # MV
mom = 450 # GeV/c
bucketArea, bunchArea,xbucket,Ubucket,x_bunch,U_bunch,x_phaseSpace,E_phaseSpace,x_bunchPhaseSpace,E_bunchPhaseSpace = calculateEmittance(rf_voltage_main,1,0.,mom*1e3,bunchLength=1.2e-9)  
x_phaseSpace*=1e9/(2*np.pi*400e6)
E_axis_ML = np.arange(-50,50)*10e6
E_axis_tomo = np.arange(-50,50)*6.6850E+06

for turn_id in tomoed_turns:
    preds, _ = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0), tf.expand_dims(normalizeTurn(turn_id), axis=0))
    tmp_predicted_PS = unnormalizeIMG(preds[0,:,:,0])*test_B_normFactor    
    
    f,ax = plt.subplots(1,2,num='phase space')
    ax[0].imshow(real_tomo_by_turn[turn_id],aspect='auto', vmin=0, vmax =test_B_normFactor, extent=(0,100, -50*6.6850E+06, 50*6.6850E+06), origin = 'lower',cmap=cmap_white_blue_red); ax[0].set_title('Tomo - Turn - {}'.format(turn_id))
    ax[0].plot(x_phaseSpace/0.025,E_phaseSpace,color='r')
    ax[0].set_ylim(-334e6, 334e6)
    ax[1].imshow(tmp_predicted_PS[14:114,14:114],aspect='auto', vmin=0, vmax =test_B_normFactor, extent=(0,100, -50*10E+06, 50*10E+06), origin = 'lower',cmap=cmap_white_blue_red); ax[1].set_title('ML - Turn - {}'.format(turn_id))
    ax[1].plot(x_phaseSpace/0.025,E_phaseSpace,color='r')
    ax[1].set_ylim(-334e6, 334e6)
    image_folder = r'\\afs\cern.ch\work\t\teoarg\python\LHC\protons\tomography\images_tomoVsML'
    fign =  os.path.realpath(image_folder +'/Turn_{:d}.png'.format(int(turn_id)))
    plt.savefig(fign,bbox_inches='tight')  
    f.clf()

    
#%% TRYING BIG MODEL WITH NEW DATA different voltage


# load massaged data
dataRun2_dir='G:\\Departments\\BE\\Groups\\OP\\Sections\\LHC\\George_Theo\\REAL_DATA_Run2'

centroid_offset=0
TimgData_ForModel = getTimgForModelFromDataFile(os.path.join(dataRun2_dir, 'PROFILE_B2_b20290_20180930122142.npy'), 1.2e11,
                                                test_T_normFactor,
                                                IMG_OUTPUT_SIZE, zeropad, start_turn, skipturns,
                                                centroid_offset=centroid_offset)
plt.figure();plt.imshow(TimgData_ForModel[:,:,0]); plt.colorbar()
#predict first turn
turn_id = 0.0
predicted_PS, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0),
                                      tf.expand_dims(normalizeTurn(turn_id), axis=0))
extracted_parameters = unnormalize_params(*list(parss[0].numpy()))
predicted_PS = unnormalizeIMG(predicted_PS[0,:,:,0])*test_B_normFactor

plt.figure();
plt.imshow(predicted_PS, origin='lower'); plt.colorbar();plt.title('Turn {}'.format(turn_id))
print(extracted_parameters)    



    
predicted_T_profiles={}
# Predict evolution for 300 turns
for turn_id in np.arange (0, 300):
  predicted_PS, parss = eCED.predictPS(tf.expand_dims(TimgData_ForModel, axis=0), tf.expand_dims(normalizeTurn(turn_id), axis=0))
  predicted_PS=unnormalizeIMG(predicted_PS[0,:,:,0])*test_B_normFactor
  predicted_T_profiles[turn_id] = np.sum(predicted_PS,0)


# compare time profiles with your tomo
timeScale_for_tomo, real_BunchProfiles =getTimeProfiles_FromData(os.path.join(dataRun2_dir, 'PROFILE_B2_b20290_20180930122142.npy'), 1.2e11)

TFs=[]
for turn_id in np.arange(0,300, 1):
    plt.figure()
    # plt.plot(real_BunchProfiles[:,turn_id], label ='real')
    # plt.plot(predicted_T_profiles[turn_id][14:114], label ='Predicted')
    plt.plot(normalize(real_BunchProfiles[:,turn_id]), label ='real')
    plt.plot(normalize(predicted_T_profiles[turn_id][14:114]), label ='Predicted')
    plt.title('Turn {}'.format(turn_id))
    plt.legend()
    
    

