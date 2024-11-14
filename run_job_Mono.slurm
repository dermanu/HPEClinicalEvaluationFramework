#!/bin/bash
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-00:15:00          # Adjust time as needed
#SBATCH --nodes=1                   # 1 compute node
#SBATCH --cpus-per-task=8           # 2 CPU cores
#SBATCH --mem=64G                  # Adjust memory if needed
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%j.log     # Save logs with job name and ID

# Load required modules
module purge
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0
module PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module torchvision/0.16.0-foss-2023a-CUDA-12.1.1
source ~/cluster/home/emanueal/EVAL/bin/activate

# Display job details
echo "Job directory: $SLURM_SUBMIT_DIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CPU cores per node: $SLURM_CPUS_ON_NODE"

# Run the video processing script
cd /cluster/home/emanueal/Documents/HPEClinicalEvaluationFramework

wandb agent dermanu/HPE_framework_3/se9s48z0 #Mono