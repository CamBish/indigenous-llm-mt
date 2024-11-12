#!/bin/bash
#SBATCH --job-name=simple_test
#SBATCH --time=0-0:15:00


cd $project/indigenous-llm-mt
module purge
module load python/3.11

cd src

python simple_test.py