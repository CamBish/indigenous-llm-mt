#!/bin/bash
#SBATCH --job-name=simple_test        # Job name
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --ntasks-per-node=2           # Number of tasks per node
#SBATCH --mem=8G
#SBATCH --time=0-0:20:00               # Maximum execution time (HH:MM:SS)
#SBATCH --account=rrg-zhu2048
#SBATCH --mail-user=cam.t.bishop@gmail.com
#SBATCH --mail-type=ALL

cd ~/$projects/indigenous-ll-mt
module purge
module load python/3.11

pip install --upgrade pip --no-index

virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

cd src/scripts

python simple_test.py