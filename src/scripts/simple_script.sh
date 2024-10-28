#!/bin/bash
#SBATCH --job-name=simple_test
#SBATCH --time=0-0:15:00


cd ~/$projects/indigenous-ll-mt
module purge
module load python/3.11

cd src/scripts

python simple_test.py