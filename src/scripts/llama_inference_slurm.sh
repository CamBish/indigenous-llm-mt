#!/bin/bash
#SBATCH --job-name=train_llm          # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=1           # Number of tasks per node
#SBATCH --cpus-per-task=32            # Number of CPU cores per task
#SBATCH --gres=gpu:1                  # Number of GPUs per node
#SBATCH --mem=8G
#SBATCH --time=0-0:20:00               # Maximum execution time (HH:MM:SS)
#SBATCH --output=./slurm_out/test_llama_inference-%j.out            
#SBATCH --error=./slurm_out/test_llama_inference-%j.err
#SBATCH --account=rrg-zhu2048