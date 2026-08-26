#!/bin/bash
#SBATCH --account=teta
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=gpu-h100s
#SBATCH --mail-user=kinjal.govil@nlr.gov
#SBATCH --mail-type=ALL
#SBATCH --chdir=/projects/teta/kgovil/mobilitynet-analysis-scripts
#SBATCH --output=/projects/teta/kgovil/mobilitynet-analysis-scripts/datasets/gae_training/loss.log

source setup/activate_conda.sh

set -euo pipefail

conda run --no-capture-output python ML/train_gae.py --batch-size 128 --epochs 3000

