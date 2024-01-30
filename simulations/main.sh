#!/bin/bash
# export OMP_NUM_THREADS=4
source $HOME/.bashrc

cd $HOME/work/git/tomography_ML/simulations

ulimit -c 0

export PYTHONPATH="$HOME/work/git/tomography_ML/simulations/BLonD:$PYTHONPATH"
/afs/cern.ch/work/k/kiliakis/install/anaconda3/bin/python sim_flatbottom_tomo.py $1

