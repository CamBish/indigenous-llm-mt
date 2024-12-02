#%%
# import argparse
import os

from typing import Literal, get_args

# from IPython.display import display

from utils import SPLITS
from utils import serialize_parallel_corpus, serialize_gold_standards, load_inuktitut_parallel_corpus, inuktitut_process_and_filter, load_cree_parallel_data, fix_cree_punctuation

DATA_MODES = Literal['parallel_corpus', 'gold_standard']

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")


INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "preprocessed",
    "inuktitut-syllabic",
    "tc"
)

INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir,
    "data",
    "preprocessed",
    "inuktitut-romanized",
    "tc"
)

GOLD_STANDARD_PATH = os.path.join(
    project_dir, "data", "external", "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0", "gold-standard"
)

CREE_PATH = os.path.join(
    project_dir, "data", "preprocessed"
)

SERIALIZED_GOLD_STANDARD_PATH = os.path.join(
    project_dir, "data", "serialized", "gold_standard.parquet"
)

SERIALIZED_CREE_PATH = os.path.join(
    project_dir, "data", "serialized", "cree_corpus.parquet"
)

GOLD_STANDARD_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "gold-standard",
)

#%%
if __name__ == '__main__':
    # Serialize all splits of Inuktitut data, for both Roman and Syllabic scripts
    for split in get_args(SPLITS):

        # Serialize and preprocess Syllabic Inuktitut
        SERIALIZED_INUKTITUT_SYLLABIC_PATH = os.path.join(
            project_dir, "data", "serialized", f"{split}_syllabic_parallel_corpus.parquet"
        )
        syllabic_parallel_corpus = load_inuktitut_parallel_corpus(INUKTITUT_SYLLABIC_PATH, split=split)
        syllabic_parallel_corpus = inuktitut_process_and_filter(syllabic_parallel_corpus)
        syllabic_parallel_corpus.to_parquet(SERIALIZED_INUKTITUT_SYLLABIC_PATH)

        # Serialize and preprocess Romanized Inuktitut
        SERIALIZED_INUKTITUT_ROMAN_PATH = os.path.join(
            project_dir, "data", "serialized", f"{split}_roman_parallel_corpus.parquet"
        )
        roman_parallel_corpus = load_inuktitut_parallel_corpus(INUKTITUT_ROMAN_PATH, split=split)
        roman_parallel_corpus = inuktitut_process_and_filter(roman_parallel_corpus)
        roman_parallel_corpus.to_parquet(SERIALIZED_INUKTITUT_ROMAN_PATH)

    cree_parallel_corpus = load_cree_parallel_data(CREE_PATH)
    cree_parallel_corpus = cree_parallel_corpus.applymap(fix_cree_punctuation)
    print(cree_parallel_corpus)
    cree_parallel_corpus.to_parquet(SERIALIZED_CREE_PATH)

    # serialize_parallel_corpus(
    #     input_path=SERIALIZED_CREE_PATH,
    #     output_path=SERIALIZED_CREE_PATH,
    #     mode='cree',
    # )

    # serialize_gold_standards(
    #     input_path=GOLD_STANDARD_PATH, output_path=SERIALIZED_GOLD_STANDARD_PATH
    # )
# %%