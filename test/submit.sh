#!/bin/bash
#COBALT -n 12
#COBALT -t 610
#COBALT -q full-node
#COBALT -A CVD-Mol-AI
#COBALT --attrs pubnet=true:enable_ssh=1:ssds=required:ssd_size=2048

source /lus/theta-fs0/projects/RL-fold/msalim/SC20-GB-CS1-ThetaGPU-AI-driven-MD/env/bin/activate

python -m deepdrivemd.driver.experiment_main $1
