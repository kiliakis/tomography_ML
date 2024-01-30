# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 14:05:30 2020

@author: teoarg
"""
# from __future__ import division
# import yaml
# import numpy as np

# f_bl = 1e-9
# f_int = 1e11

# N_b_arr = np.linspace(1.1,1.3,3)
# bunch_length_arr = np.linspace(1.3,1.6,4)
# phase_error_arr = np.linspace(-30,30,10)
# energy_error_arr = np.linspace(-70,70,10)


# #N_b_arr = np.linspace(1.1,1.3,1)
# #bunch_length_arr = np.linspace(1.3,1.6,1)
# #phase_error_arr = np.linspace(-30,30,4)
# #energy_error_arr = np.linspace(-70,70,4)

# counter = 0
# for N_b in N_b_arr:
#     for bunch_length in bunch_length_arr:
#         for phase_error in phase_error_arr:
#             for energy_error in energy_error_arr:
#                 counter +=1
#                 config_file = r'./input/phEr%f_enEr%f_bl%1.2e_int%1.2e.yaml'%(phase_error,energy_error,bunch_length*f_bl,N_b*f_int) 
#                 beam_params = {}
#                 beam_params['N_b'] = int(N_b*f_int)
#                 beam_params['bunch_length'] = float(bunch_length*f_bl)
#                 beam_params['phError'] = int(phase_error)
#                 beam_params['enError'] = int(energy_error)
# #                with open(config_file,'w') as outputfile:
# #                    beam_params = yaml.dump(beam_params,outputfile,default_flow_style=False)
# #                outputfile.close()
                
# print('{:d} jobs to submit)'.format(counter))
    
#%%
from __future__ import division
import yaml
import numpy as np
import random
f_bl = 1e-9
f_int = 1e11

#random.seed(1000)
N_b_arr = np.array([])
bunch_length_arr = np.array([])
N_b_arr = np.array([])
phase_error_arr = np.array([])
energy_error_arr = np.array([])
Vrf_arr = np.array([])
mu_arr = np.array([])
VrfSPS_arr = np.array([])
# ZoN_arr = np.array([])

for x in range(20000):
    N_b = round(random.uniform(0.1,3.0),2)
    N_b_arr = np.append(N_b_arr,N_b)
    bunch_length = round(random.uniform(1.2,1.8),2)
    bunch_length_arr = np.append(bunch_length_arr,bunch_length)
    phase_error = random.randint(-50,50)
    phase_error_arr = np.append(phase_error_arr,phase_error)
    energy_error = random.randint(-100,100)
    energy_error_arr = np.append(energy_error_arr,energy_error)
    Vrf = round(random.uniform(3.0,9.2),1)
    Vrf_arr = np.append(Vrf_arr,Vrf)
    VrfSPS = round(random.uniform(5.0,12.0),1)
    VrfSPS_arr = np.append(Vrf_arr,Vrf)
    mu = round(random.uniform(1.0,5.0),1)
    mu_arr = np.append(mu_arr,mu)
    # ZoN = round(random.uniform(0.05,0.10),3)
    # ZoN_arr = np.append(ZoN_arr,ZoN)
    
    
    config_file = r'./input/phEr%i_enEr%i_bl%1.2e_int%1.2e_Vrf%1.1f_mu%1.1f_VrfSPS%1.1f.yaml'%(phase_error,energy_error,bunch_length*f_bl,N_b*f_int,Vrf,mu,VrfSPS) 
    print(config_file)
    beam_params = {}
    beam_params['N_b'] = int(N_b*f_int)
    beam_params['bunch_length'] = float(bunch_length*f_bl)
    beam_params['phError'] = int(phase_error)
    beam_params['enError'] = int(energy_error)
    beam_params['Vrf'] = float(Vrf)
    beam_params['VrfSPS'] = float(VrfSPS)
    beam_params['mu'] = float(mu)
    # beam_params['ZoN'] = float(ZoN)

    with open(config_file,'w') as outputfile:
        beam_params = yaml.dump(beam_params,outputfile,default_flow_style=False)
    outputfile.close()

