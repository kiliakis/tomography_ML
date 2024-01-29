# Train, Optimize, and Evaluate ML Longitudinal Tomography in the LHC

## General purpose
The purpose of this project is to leverage the longitudinal bunch profiles, to extract several beam features at injection in the LHC using Machine learning. 

These features include:
* Phase and Energy injection errors
* Bunch length and Intensity
* Beam observed RF Voltage in the LHC and in the SPS
* The mu parameter of the Gaussian shape particle distribution in the SPS

In addition to these features, a separate model can be used to reconstruct the longitudinal phase-space. 


## Presentations, posters, and papers
Links

## 1. Synthetic data Generation
The training data of the model are generated using the BLonD simulator. The mainfile used can be found in simulations directory.
Directory contents:
* sim_flatbottom_tomo.py: Simulation mainfile. LHC flatbottom with intensity effects, single-bunch. Bunch generated and matched in SPS. 
* prepare_sim_tomo.py: Script that generates the design space. You can specify the parameters to be scanned, the ranges for each parameter, and the total number of input files to be generated. 
* main.sh: The bash script that will be run by the job execution node. 
* htcondor.conf: HTCondor configuration script.
* run.sh: This will launch one job per generated input file by the prepare_sim_tomo.py script.


## 2. Synthetic data pre-processing
The set of scripts generate_encoder_data.py, generate_decoder_data.py and generated_tomoscope_data.py is used to format the simulations output data
in a format ready to be used for training. 
Scripts: 
* generate_encoder_data.py: For every simulation output, one pickle file will be created that will contain the configuration parameters, together with the longitudinal bunch profiles. The bunch profiles can be optionally convolved with a transfer function, to better match the real measurement data. 
* generate_decoder_data.py: For every simulation output, a number (--turns-per-case) of pickle files will be created that will contain the configuration parameters, the bunch profiles, and the phase-space at a given turn. 
* generate_tomoscope_data.py: For every simulatoin output, one pickle file will be created, that will contain the config params, the bunch profiles, and a number (--turns-per-case) of phase-space turns. 

Reading multiple small files is slow compared to reading a small number of larger files.
This is why in addition to the generate_encoder/decoder/tomoscope_data.py scripts one case use the merge_data_single_file.py
script that will merge a large number (modifiable, 5k by default) of pickle files to a single file, in order to accelerate the dataset loading during training/validation.


## 3. Inspecting the input data
A set of scripts that can be used to inspect the generated data. Can be useful to understand
various properties of the training datasets. 

## 3. Model Design
The model architectures can be found in models.py
More specifically, that available models are the following:
    
    1. AutoEncoderSkipAhead
    1. VariationalAutoEncoder
    2. AutoEncoderEfficientNet
    3. AutoEncoderTranspose
    4. FeatureExtractor
    5. EncoderSingleViT
    6. EncoderSingle
    7. EncoderMulti
    8. Decoder
    9. EncoderDecoderModel
    10. EncoderOld
    11. TomoscopeOld
    12. Tomoscope

## 4. Training
There are several scripts that can be used for training the models.
For interactive training, suggested for debugging purposes:

For batch submission:

## 5. Hyper-parameter optimization
A set of scripts is provided that can be used for hyper-parameter optimization of the various models.
The optuna framework is used to guide the search space exploration. 

## 6. Evaluation on synthetic data
To evaluate the trained models against the validation dataset, the following scripts are used:


## 7. Evaluation on real data
To evaluate the model on measurement data, the following scripts are provided:

## 9. Other utilities
Visualize the weights of the model: 


## Original Authors
T. Argyropoylos
G. Trad
K. Iliakis
H. Timko