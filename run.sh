#!/bin/bash

# Copyright
# 2024, Johns Hopkins University (Author: Prabhav Singh)
# Apache 2.0.

#SBATCH --job-name=DataExploration
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=16GB
#SBATCH --partition=cpu
#SBATCH --mail-user="psingh54@jhu.edu"

source /home/psingh54/.bashrc
module load cuda/11.7

conda activate avd-hyperion

# NORMALIZE, 4 FFN
python /export/fs06/psingh54/PseudoSpeaker/analyze_capspeech.py