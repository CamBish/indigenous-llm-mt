# import argparse
import os

from typing import Literal

from utils import SPLITS
from utils import serialize_parallel_corpus, serialize_gold_standards

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

if __name__ == '__main__':
    # Serialize all splits of Inuktitut data, for both Roman and Syllabic scripts
    for split in SPLITS:
        SERIALIZED_INUKTITUT_SYLLABIC_PATH = os.path.join(
            project_dir, "data", "serialized", f"{split}_syllabic_parallel_corpus.parquet"
        )
        serialize_parallel_corpus(
            input_path=INUKTITUT_SYLLABIC_PATH,
            output_path=SERIALIZED_INUKTITUT_SYLLABIC_PATH,
            split=split,
            mode='inuktitut',
        )

        SERIALIZED_INUKTITUT_ROMAN_PATH = os.path.join(
            project_dir, "data", "serialized", f"{split}_roman_parallel_corpus.parquet"
        )
        serialize_parallel_corpus(
            input_path=INUKTITUT_ROMAN_PATH,
            output_path=SERIALIZED_INUKTITUT_ROMAN_PATH,
            split=split,
            mode='inuktitut',
        )

    serialize_parallel_corpus(
        input_path=SERIALIZED_CREE_PATH,
        output_path=SERIALIZED_CREE_PATH,
        mode='cree',
    )

    serialize_gold_standards(
        input_path=GOLD_STANDARD_PATH, output_path=SERIALIZED_GOLD_STANDARD_PATH
    )