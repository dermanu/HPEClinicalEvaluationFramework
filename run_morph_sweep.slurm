#!/bin/bash
#SBATCH --job-name=morph_synth
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=morph_%j.log

export PYTHONNOUSERSITE=1
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /cluster/home/emanueal/morphenv/bin/activate
cd /cluster/home/emanueal/Documents/HPEClinicalEvaluationFramework/HPEClinicalEvaluationFramework

python -c "import torch; print('[device check] cuda available:', torch.cuda.is_available())"

# SMOKE first (fast). Comment out once it looks right.
python -m skeletonMorphing.trainSyntheticMorphing --arrays synth_arrays \
  --variants pure --regimes linear --holdout 25 26 --epochs 3

# FULL sweep — uncomment, bump --time to 04:00:00, resubmit after smoke passes.
# python -m skeletonMorphing.trainSyntheticMorphing --arrays synth_arrays --holdout 25 26