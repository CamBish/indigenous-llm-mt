#!/bin/bash

module load StdEnv/2023
ml cuda python/3.11 arrow/17.0.0 gcc

virtualenv --no-download $SLURM_TMPDIR/ENV

source $SLURM_TMPDIR/ENV/bin/activate

pip install --upgrade pip --no-index

pip install --no-index torch scikit_learn tqdm nltk torchtext transformers>=4.43.1 spacy triton accelerate datasets scipy matplotlib numpy huggingface_hub ipython

echo "Done installing virtualenv!"