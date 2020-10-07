#!/bin/bash
#COBALT -n 2 
#COBALT -t 120 
#COBALT -q R.CVD-Mol-AI 
#COBALT -A CVD-Mol-AI 
#COBALT --attrs pubnet=true:enable_ssh=1:ssds=required:ssd_size=2048

source /lus/theta-fs0/projects/RL-fold/msalim/SC20-GB-CS1-ThetaGPU-AI-driven-MD/env/bin/activate
mpirun -n 2 -npernode 1 rm -r /raid/scratch/*
rm -r /home/msalim/RL-fold/msalim/test-mdonly

python -m deepdrivemd.driver.experiment_main /home/msalim/RL-fold/msalim/SC20-GB-CS1-ThetaGPU-AI-driven-MD/examples/test-md-only.yaml
