#!/bin/bash

# Define the study folder path and participant folders
participant_folders=("par1" "par10" "par11" "par2" "par3" "par4" "par5" "par6" "par7")  # Add more as needed

# Submit a job for each participant folder
for participant in "${participant_folders[@]}"; do
  sbatch --export=participant="$participant" run_job.slurm
done
