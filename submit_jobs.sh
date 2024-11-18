#!/bin/bash

# Run `run_job_Mono.slurm` twice
for ((i=1; i<=10; i++)); do
  sbatch run_job_Mono.slurm
done

# Run `run_job_Multi.slurm` twice
for ((j=1; j<=10; j++)); do
  sbatch run_job_Multi.slurm
done
