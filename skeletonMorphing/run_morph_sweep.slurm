#!/bin/bash
#SBATCH --job-name=morph_synth
#SBATCH --account=YOUR_ACCOUNT          # your allocation/project code
#SBATCH --partition=GPUQ                # confirm the current GPU partition name
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=morph_%j.log

module purge
module load Python/3.10                 # match what you built the env with
source ~/morphenv/bin/activate

cd ~/HPEClinicalEvaluationFramework

# (optional) re-extract on the cluster for a self-contained run:
# python -m skeletonMorphing.extractGTHPE --data_path . --out synth_arrays

# full sweep: both variants x three regimes, held-out 25 & 26
python -m skeletonMorphing.trainSyntheticMorphing --arrays synth_arrays --holdout 25 26