#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-48:00:00          # Adjust time as needed
#SBATCH --nodes=1                   # 1 compute node
#SBATCH --cpus-per-task=2           # 2 CPU cores
#SBATCH --mem=32G                  # Adjust memory if needed
#SBATCH --output=logs/%x_%j.log     # Save logs with job name and ID
#SBATCH --mail-user=emanueal@ntnu.no
#SBATCH --mail-type=ALL

# Load required modules
module purge
module load GCCcore/13.3.0
module swap GCCcore/13.3.0 GCCcore/12.3.0
module load Python/3.11.3-GCCcore-12.3.0
module load FFmpeg/6.0-GCCcore-12.3.0
module PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
source ~/cluster/home/emanueal/HPE/bin/activate

# Display job details
echo "Job directory: $SLURM_SUBMIT_DIR"
echo "Job name: $SLURM_JOB_NAME"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CPU cores per node: $SLURM_CPUS_ON_NODE"

# Define the participant's video path
video_path="/cluster/home/emanueal/Desktop/MoCap/segmented/"

# Run the video processing script
cd /cluster/home/emanueal/Documents/HPEClinicalEvaluationFramework
python evaluation_pipeline.py "$video_path"

deactivate