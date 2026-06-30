cat > /cluster/home/emanueal/Documents/HPEClinicalEvaluationFramework/sweep.slurm <<'EOF'
#!/bin/bash
#SBATCH --job-name=morph_sweep
#SBATCH --account=share-ie-idi
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=sweep_%j.log

export PYTHONNOUSERSITE=1
module purge
module load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source /cluster/home/emanueal/morphenv/bin/activate
cd /cluster/home/emanueal/Documents/HPEClinicalEvaluationFramework

python -c "import torch; print('[device check] cuda available:', torch.cuda.is_available())"
python -m skeletonMorphing.trainSyntheticMorphing --arrays synth_arrays --holdout 25 26
EOF
sbatch sweep.slurm
squeue --me