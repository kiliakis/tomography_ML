# -*- coding: utf-8 -*-
"""
Created on Tue May 12 14:42:46 2020
Simulations of LLD in the LHC flat bottom
@author: teoarg
"""

from __future__ import division
from blond.trackers.tracker import RingAndRFTracker, FullRingAndRF
from blond.plots.plot import Plot
import time
t_initial=time.perf_counter()
import numpy as np

import os
import h5py
import yaml
import argparse
import matplotlib.pyplot as plt

# BLonD imports
from blond.input_parameters.ring import Ring #, RingOptions
from blond.input_parameters.rf_parameters import RFStation
from blond.beam.beam import Beam, Proton
from blond.beam.profile import Profile, CutOptions
from blond.beam.distributions import matched_from_distribution_function
from blond.impedances.impedance import InducedVoltageFreq, TotalInducedVoltage#, InducedVoltageTime, InductiveImpedance
from blond.impedances.impedance_sources import Resonators, InputTable
from blond.longitudinal_impedance_SPS.impedance_scenario import scenario, impedance2blond

#from blond.llrf.beam_feedback import BeamFeedback

def potentialWell(x,V):
#calculate the potential well
#    x_0 = x[0]
#    x -= x_0
    dx = x[1]-x[0]
    U = np.cumsum(V)*dx
    
#    U_max = np.argmax([np.abs(np.max(U)),np.abs(np.min(U))])
    U_max = np.argmax([np.max(U),np.min(U)])
    if U_max==0:
        U = -U        
    U = U -U.min()

    ##find the stable phase
#    fixed_points = np.where((x-x_0)<=np.mean(x)+0.2*np.mean((x-x_0)))[0]
#    fixed_points = np.where(V[fixed_points]>=0)[0]
#    phis = x[fixed_points[-1]]
    positive_points = np.where(V>=0)[0]
    fixed_points = np.where(np.diff(positive_points)>1)[0]
    if len(fixed_points)>0:
        phis = x[positive_points[fixed_points[0]]]
        U -= U[positive_points[fixed_points[0]]]
    else:
        phis = x[positive_points[-1]] 
        U -= U[positive_points[-1]]
        
    
    
#    minV = min(U)
#    phis = x[np.where(U==minV)]
    
        
    return U,phis


def ring_params(p,machine='LHC',mass=1.67e-27):
# get the parameters for the machine and the calculated values in correct units
    #constants
    c = 2.99792458e8 #m/s
#    m = 1.67e-27 # proton mass in kgr
#    m = 207.2*1.67e-27 # Pb mass in kgr
    E0 = ((mass*c**2)/1.6e-10)*1e3 #in MeV
#    print(E0)

    if machine == 'SPS':
        R = 1100.0093 # radius in m
        bending_radius=741.254598153917 # in m
        gammat = 18 #22.93662195  #22.8 #18 # transision gamma
        h = 4620 # harmonic number
    elif machine == 'LHC':
        R = 4242.89 # radius in m
        bending_radius=2804.0 # in m        
        gammat = 55.76 #53.61 # transision gamma 55.744024 b1, 55.852493 b2@injection
        h = 35640 # harmonic number
    elif machine == 'LEIR':
        R = 78.54370266/2/np.pi # radius in m
        bending_radius=30.0 # not correct       
        gammat = 2.83744
        h = 2 # harmonic number
    else:
        print('I do not have the parameters for the machine you request')
    
#calculated parameters
    E = np.sqrt(p**2 + E0**2) #in MeV
    gamma = E/E0
    beta = np.sqrt(1-(1/gamma**2))
    v = beta*c
    Trev = 2*np.pi*R/v # in s
    wrev = 2*np.pi/Trev # in radHz
    wrf = h*wrev # in rad Hz
    frf = wrf/2/np.pi # in Hz
    eta = (1./gammat)**2-(1./gamma)**2
    
    # normalization factor
    emitScalingFactor = np.sqrt((2*beta**2*E)/(h*np.abs(eta)*wrev**2*2*np.pi))/h
    #this factor has to be multiplied by the DE (MeV) of the particles
    HamiltScalingFactor = (h*eta*2*np.pi)/(beta**2*E)
    
    params = {}
    params['R'] = R
    params['gammat'] = gammat
    params['bending_radius'] = bending_radius       
    params['emitScalingFactor'] = emitScalingFactor
    params['HamiltScalingFactor'] = HamiltScalingFactor
    params['E'] = E
    params['gamma'] = gamma
    params['beta'] = beta
    params['Trev'] = Trev
    params['wrev'] = wrev
    params['wrf'] = wrf
    params['frf'] = frf
    params['eta'] = eta
    params['h'] = h 
    params['E0'] = E0    
                               
    return params


def coord2Ham_blond(x,totalVoltage,beam_dt,beam_dE,p=450e3,Vs=0,machine='LHC',mass=1.67e-27):
    
    
    x_new = np.linspace(x[0],x[-1],2**15)
    Vtot = np.interp(x_new,x,totalVoltage)-Vs
    x = x_new
    U,phis = potentialWell(x,Vtot) #potential well
    
    params = ring_params(p,machine,mass)
    HamiltScalingFactor = params['HamiltScalingFactor']
   
    Ham = HamiltScalingFactor*((beam_dE**2)/2) + np.interp(beam_dt,x,U)
    return Ham

parser = argparse.ArgumentParser(description='Run main file,\
                                     expect a configuration file')
parser.add_argument('configfile', type=str, nargs=1, 
                    default='./input/default.yaml')
args = parser.parse_args()

Filename = args.configfile[0]

with open(Filename,'r') as inputfile:
    beam_params = yaml.safe_load(inputfile)

bunch_length = float(beam_params['bunch_length'])#2.5e-9
intensity = int(beam_params['N_b'])
phaseKick = True
offset = float(beam_params['phError']) # [deg]
dt_err = offset*1.25e-9/180.  # Injection error 0.07
energyKick = True
energyOffset = float(beam_params['enError']) # in MeV
dE_err = energyOffset*1e6  # in eV
Vrf = float(beam_params['Vrf']) # in MV
mu_binomial = float(beam_params['mu']) 
VrfSPS = float(beam_params['VrfSPS'])
ZoN = 0.076 #float(beam_params['ZoN']) 

# Output folder
#filename = 'test_LHC_LLD.pkl'
output_folder = r'./results_tomo/phEr%d_enEr%d_bl%1.2e_int%1.2e_Vrf%1.1f_mu%1.1f_VrfSPS%1.1f'%(offset,energyOffset,bunch_length,intensity,Vrf,mu_binomial,VrfSPS)
                

#output_folder = r'../results_tomo/phEr%d_enEr%d_bl%1.2e_int%1.2e'%(2.,30.,1.3e-9,1.0e11)
try:
    os.mkdir(output_folder)
except:
    pass

print('Filename = ',Filename)
print('phase offset = {}'.format(offset))
print('energy offset = {}'.format(energyOffset))
print('Bunch length = {}'.format(bunch_length))
print('Bunch intensity = {}'.format(intensity))
print('Vrf = {}'.format(Vrf))
print('VrfSPS = {}'.format(VrfSPS))
print('mu = {}'.format(mu_binomial))



# Simulation parameters
n_macroparticles = int(1e6)
#plot and save every MPLOT turns
MPLOT = 200 # only for printing 
n_turns = 500

##%% generate the beam in the SPS flat top
# General parameters SPS
particle_type = Proton()
circumference = 6911.5038                       # Machine circumference [m]
gamma_transition = 17.95142852 #22.93662195 #22.8 #17.95142852                  # Transition gamma
momentum_compaction = 1./gamma_transition**2    # Momentum compaction array

# RF parameters
n_rf_systems = 2
Vr = 0.18
mode = 'BSM'
harmonic_number = 4620.
phi_rf = 0.

# Profile parameters
n_slices = 2**8 #2**8 #2**8 #64 correspond to Nyquist frequency 6.4 GHz # 2**8
cut_left = 0 - 0.3
cut_right = 2*np.pi + 0.3

distribution_type = 'binomial' #'parabolic_amplitude'
mu = mu_binomial #2.0 #1.5
seed = 1234

####
modelStr = 'futurePostLS2' #'present2018'  # or 'futurePostLS2'

### define the ring
momentum = 450.0e+9 # eV
ring = Ring(circumference, momentum_compaction,
            momentum, particle_type,1)

rf_voltage = VrfSPS*1e6  
rf_station = RFStation(ring, [harmonic_number,4*harmonic_number],
                             [rf_voltage,Vr*rf_voltage],
                             [phi_rf,np.pi],n_rf=n_rf_systems)
                                
### Beam
SPS_beam = Beam(ring, n_macroparticles, intensity)

### Profile
cut_options = CutOptions(cut_left, cut_right, n_slices=n_slices,
                            cuts_unit='rad', RFSectionParameters=rf_station)
profile = Profile(SPS_beam, CutOptions=cut_options)

##%% load the SPS impedance model
frequencyResolution = ring.f_rev[0]*2  # Hz; 44kHz ~ f_rev
# create scenario; can be any .yaml file that stores the elements
impScenario = scenario(modelStr+'_SPS.txt')
#impScenario = scenario(modelStr+'_SPS_noMain200TWC.txt')

# convert scenario into BLonD model
impModel = impedance2blond(impScenario.table_impedance)
# create object to calculate induced voltage via the 'frequency' method
imp_list = impModel.impedanceList

# INDUCED VOLTAGE FROM IMPEDANCE ----------------------------------------------
SPSimpedance = InducedVoltageFreq(SPS_beam, profile, imp_list,
                                   frequency_resolution=frequencyResolution)

SPSimpedance_table = InputTable(SPSimpedance.freq,SPSimpedance.total_impedance.real*profile.bin_size,SPSimpedance.total_impedance.imag*profile.bin_size)

# Total impedance
impedance_freq = InducedVoltageFreq(SPS_beam, profile, [SPSimpedance_table],
                                   frequency_resolution=frequencyResolution)
total_ind_volt = TotalInducedVoltage(SPS_beam, profile, [impedance_freq])


longitudinal_tracker = RingAndRFTracker(rf_station, SPS_beam,
                                        BeamFeedback=None,
                                        Profile=profile,
                                        TotalInducedVoltage=total_ind_volt,
                                        solver='simple',
                                        interpolation=True)    
full_tracker = FullRingAndRF([longitudinal_tracker])

# Bunch generation
resultGeneration = matched_from_distribution_function(SPS_beam, full_tracker,
                               distribution_function_input=None,
                               distribution_user_table=None,
                               main_harmonic_option='lowest_freq',
                               TotalInducedVoltage=total_ind_volt,
                               n_iterations=20, n_points_potential=n_slices*2,
                               n_points_grid=n_slices*2,
                               dt_margin_percent=0.01,
                               extraVoltageDict=None, seed=seed,
                               distribution_exponent=mu,
                               distribution_type='binomial',
                               emittance=None, bunch_length=bunch_length,
                               bunch_length_fit='fwhm',
                               distribution_variable='Hamiltonian',
                               process_pot_well = False,
                               turn_number=0)

profile.track()
total_ind_volt.induced_voltage_sum()
profile.fwhm()

x = SPS_beam.dt
y = SPS_beam.dE

heatmap, xedges, yedges = np.histogram2d(x, y, bins=[100,100])
plt.imshow(heatmap.T, origin='lower', cmap='jet')
# plt.colorbar(label='Number of particles')
# Add labels and title
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('Bunch Distribution')
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('bunch_distribution.png', dpi=300, bbox_inches='tight')
plt.show()
# format_options = {'dirname': './'}
# plots = Plot(ring, rf_station, SPS_beam, MPLOT, n_turns, 0, 2*np.pi,
#              -350e6, 350e6, xunit='rad', separatrix_plot=True, histograms_plot=True,
#              Profile=profile, format_options=format_options)
# plots.track()
exit()

# put the beam in the center of the LHC bucket
sps_phis = np.mean(SPS_beam.dt)
SPS_beam.dt += (-sps_phis + 1.25e-9)

##%% Now inject this beam in the LHC
# General parameters LHC
particle_type = Proton()
circumference = 26658.883 # Machine circumference [m]
gamma_transition = 55.759505 # Transition gamma
momentum_compaction = 1./gamma_transition**2    # Momentum compaction array

# RF parameters
n_rf_systems = 1
harmonic_number = 35640.
phi_rf = 0.

# Profile parameters
n_slices = 100 #2**8 #2**8 #64 correspond to Nyquist frequency 6.4 GHz # 2**8
cut_left = 0 #- 0.003
cut_right = 2*np.pi #+ 0.003

# impedance model to use
# ZoN = 0.09 # Ohm
f_R = 5e9 #int(beam_params['f_R'])
Q = 1

momentum = 450.0e+9 # eV
ring = Ring(circumference, momentum_compaction,
            momentum, particle_type,n_turns)

rf_voltage = Vrf*1e6 #4.0e6  
rf_station = RFStation(ring, [harmonic_number],
                             [rf_voltage],
                             [phi_rf])
                                
### Beam
beam = Beam(ring, n_macroparticles, intensity)
beam.dt = SPS_beam.dt
beam.dE = SPS_beam.dE

### Profile
cut_options = CutOptions(cut_left, cut_right, n_slices=n_slices,
                            cuts_unit='rad', RFSectionParameters=rf_station)
profile = Profile(beam, CutOptions=cut_options)

#Impedance
R_sh = f_R*ZoN*Q/ring.f_rev[0]
resonator = Resonators([R_sh], [f_R], [Q], method='c++')
n_freq = 10000
frequencyResolution = n_slices/rf_station.t_rf[0,0]/n_freq #rf_st ring.f_rev[0]*2  # Hz; 44kHz ~ f_rev

# INDUCED VOLTAGE FROM IMPEDANCE-----------------------------------------------
LHCimpedance = InducedVoltageFreq(beam, profile, [resonator],
                    frequency_resolution=frequencyResolution)
#LHCimpedance = InducedVoltageTime(beam,profile,[resonator])
total_ind_volt = TotalInducedVoltage(beam, profile, [LHCimpedance])

# RF tracker
longitudinal_tracker = RingAndRFTracker(rf_station, beam,
                                        BeamFeedback=None,
                                        Profile=profile,
                                        TotalInducedVoltage=total_ind_volt,
                                        solver='simple',
                                        interpolation=True)    
#full_tracker = FullRingAndRF([longitudinal_tracker])
profile.track()
total_ind_volt.induced_voltage_sum()
profile.fwhm()

bunch_position_save = np.zeros(ring.n_turns+1)
bunch_position_save[0] = profile.bunchPosition
bunch_length_save = np.zeros(ring.n_turns+1)
bunch_length_save[0] = profile.bunchLength

BunchProfiles = np.zeros((len(profile.bin_centers),n_turns+2),dtype=np.uint32)
#BunchProfiles[:,0] = profile.bin_centers
BunchProfiles[:,0] = profile.n_macroparticles
 
bunch_intensity_save = np.zeros(ring.n_turns+1)
bunch_intensity_save[0] = np.trapz(profile.n_macroparticles)

EnergyProfiles = np.zeros((len(profile.bin_centers),n_turns+1),dtype=np.uint32)
#A = bm.slice(beam.dE*1e-6, energy_profile, -300,300)
energy_profile,B = np.histogram(beam.dE*1e-6,n_slices,(-500,500))
energy_profile_bin_centers = (B[0:-1]+B[1::])/2
#EnergyProfiles[:,0] = energy_profile_bin_centers
EnergyProfiles[:,0] = energy_profile

phaseSpace_density_array = np.zeros(((n_slices-1)*(n_slices-1),n_turns+1),dtype=np.uint32)
x_bin_center_array = np.zeros((n_slices-1,n_turns+1),dtype=np.float32)
y_bin_center_array = np.zeros((n_slices-1,n_turns+1),dtype=np.float32)

counts, xedges, yedges = np.histogram2d(1e9*beam.dt[::1], 1e-6*beam.dE[::1], bins=[1e9*profile.bin_centers,energy_profile_bin_centers])
x_bin_center = (xedges[0:-1]+xedges[1::])/2
y_bin_center = (yedges[0:-1]+yedges[1::])/2
phaseSpace_density_array[:,0] = np.reshape(counts,((n_slices-1)*(n_slices-1),))
x_bin_center_array[:,0] = x_bin_center
y_bin_center_array[:,0] = y_bin_center


#x = profile.bin_centers*rf_station.omega_rf[0,0]
#TotalVoltage = 1e-6*longitudinal_tracker.total_voltage
#Ham = coord2Ham_blond(x,TotalVoltage,beam.dt[::1]*rf_station.omega_rf[0,0], 1e-6*beam.dE[::1],p=momentum*1e-6,Vs=0,machine='LHC',mass=1.67e-27)
#DistrProfiles = np.zeros((len(profile.bin_centers),n_turns+1),dtype=np.uint32)
#distribution_profile,B = np.histogram(Ham,n_slices,(0,np.max(Ham)+0.2*np.max(Ham)))
#DistrProfiles[:,0] = distribution_profile

if phaseKick==True:
    beam.dt += dt_err
    
if energyKick==True:
    beam.dE += dE_err    
                                     
# Tracking ------------------------------------------------------------------
for indexTurn in range(ring.n_turns):
    
    if indexTurn%MPLOT==0:
        t0=time.perf_counter()
  
    # Track
    longitudinal_tracker.track()
    profile.track()
    profile.fwhm()
    total_ind_volt.induced_voltage_sum()    
    
    # Bunch analysis
    bunch_position_save[indexTurn+1] = profile.bunchPosition
    bunch_length_save[indexTurn+1] = profile.bunchLength
    bunch_intensity_save[indexTurn+1] = np.trapz(profile.n_macroparticles)
    BunchProfiles[:,indexTurn+1] = profile.n_macroparticles
    energy_profile,B = np.histogram(beam.dE*1e-6,n_slices,(-500,500))
    EnergyProfiles[:,indexTurn+1] = energy_profile

    counts, xedges, yedges = np.histogram2d(1e9*beam.dt[::1], 1e-6*beam.dE[::1], bins=[1e9*profile.bin_centers,energy_profile_bin_centers])
    x_bin_center = (xedges[0:-1]+xedges[1::])/2
    y_bin_center = (yedges[0:-1]+yedges[1::])/2
    phaseSpace_density_array[:,indexTurn+1] = np.reshape(counts,((n_slices-1)*(n_slices-1),))
    x_bin_center_array[:,indexTurn+1] = x_bin_center
    y_bin_center_array[:,indexTurn+1] = y_bin_center    
    
    if indexTurn%MPLOT==0:
        t1=time.perf_counter()
        print('Time per turn: {:1.3f} s'.format(t1-t0))
        print('Turn = {:d}'.format(indexTurn))
        print('Time in cycle = {:2.3f}'.format(ring.cycle_time[indexTurn]))
        print('#####################################')
        
      
#        plt.figure('Vind')
#        plt.plot(total_ind_volt.time_array,total_ind_volt.induced_voltage*1e-6,lw=2)
##            fign =  os.path.realpath(save_dir +'/Vind_Turn{:d}.png'.format(indexTurn))
#        fign =  os.path.realpath(save_dir +'/Vind.png'.format(indexTurn))
#        plt.savefig(fign)
#        
#        plt.figure('Bunch length')
#        plt.clf()
#        plt.plot(ring.cycle_time[0:indexTurn+1], 4*bunch_length_save[0:indexTurn+1]*1e9)
#        plt.xlabel('Time [s]')
#        plt.ylabel('Bunch length [ns]')
#        fign =  os.path.realpath(save_dir +'/bunchLength.png'.format(indexTurn))
#        plt.savefig(fign)
#
#
#        plt.figure('Bunch position')
#        plt.clf()
#        plt.plot(ring.cycle_time[0:indexTurn+1], bunch_position_save[0:indexTurn+1]*1e9,'b')
#        plt.xlabel('Time [s]')
#        plt.ylabel('Bunch position [ns]')
#        fign =  os.path.realpath(save_dir +'/bunchPosition.png'.format(indexTurn))
#        plt.savefig(fign)        
#
#        plt.figure('Bunch intensity')
#        plt.clf()
#        plt.plot(ring.cycle_time[0:indexTurn+1], bunch_intensity_save[0:indexTurn+1]/bunch_intensity_save[0])
#        plt.xlabel('Time [s]')
#        plt.ylabel('Bunch intensity [arb. units]')
#        fign =  os.path.realpath(save_dir +'/bunchIntensity.png'.format(indexTurn))
#        plt.savefig(fign) 
#        
#        
#        bl = bunch_length_save[indexTurn]
#        x = profile.bin_centers*rf_station.omega_rf[0,indexTurn]
#        TotalVoltage = 1e-6*longitudinal_tracker.total_voltage
#        bp = bunch_position_save[indexTurn]*1e9
#        [bucketArea, bunchArea,xbucket,Ubucket,x_bunch,U_bunch,x_phaseSpace,E_phaseSpace,x_bunchPhaseSpace,E_bunchPhaseSpace] = calculateEmittance_blond(TotalVoltage,momentum,x,bunchLength=bl,Vs=0,machine='LHC')
#        plt.figure('phase-space')
#        plt.clf()
#        plt.hexbin(1e9*beam.dt[::1], 1e-6*beam.dE[::1], gridsize=(200,200), bins='log', cmap=cmap_white_blue_red)
#        plt.plot(1e9*x_phaseSpace/rf_station.omega_rf[0,indexTurn],1e-9*E_phaseSpace,'k',
#                 1e9*x_bunchPhaseSpace/rf_station.omega_rf[0,indexTurn], 1e-9*E_bunchPhaseSpace,'r')
#        plt.xlim([np.min(1e9*x_phaseSpace/rf_station.omega_rf[0,indexTurn]),np.max(1e9*x_phaseSpace/rf_station.omega_rf[0,indexTurn])])
#        plt.xlabel('Time [ns]')
#        plt.ylabel('DE [MeV]')
#        fign =  os.path.realpath(save_dir +'/PhaseSpace_Turn{:3d}.png'.format(indexTurn))
#        plt.savefig(fign)


# save the output at the end
h5file = h5py.File(f'{output_folder}/saved_result.hdf5', 'w')
columns = h5file.create_dataset('columns', 
                (len(ring.cycle_time),4), 
                 compression="gzip", compression_opts=9, dtype=np.float64)
columns[:,0] = ring.cycle_time
columns[:,1] = bunch_length_save
columns[:,2] = bunch_position_save
columns[:,3] = bunch_intensity_save
h5file.create_dataset('bunchProfiles',data= BunchProfiles,
                 compression="gzip", compression_opts=9, dtype=np.uint32)
h5file.create_dataset('energyProfiles',data= EnergyProfiles,
                 compression="gzip", compression_opts=9, dtype=np.uint32)
h5file.create_dataset('phaseSpace_density_array',data= phaseSpace_density_array,
                 compression="gzip", compression_opts=9, dtype=np.uint32)
h5file.create_dataset('x_bin_center_array',data= x_bin_center_array,
                 compression="gzip", compression_opts=9, dtype=np.float32)
h5file.create_dataset('y_bin_center_array',data= y_bin_center_array,
                 compression="gzip", compression_opts=9, dtype=np.float32)

h5file.close()
t_final=time.perf_counter()
print('total time: {:2.4f} min'.format((t_final-t_initial)/60))
print('DONE!')
     



