#%%
# import argparse
import os
import re

from typing import Literal, get_args
from html import unescape

# from IPython.display import display
import pandas as pd

from utils import SPLITS
from utils import serialize_gold_standards, load_inuktitut_parallel_corpus, load_cree_parallel_data, get_project_root

DATA_MODES = Literal['parallel_corpus', 'gold_standard']

project_dir = get_project_root(os.path.abspath(os.path.dirname(__file__)))

# Filepath definitions
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
    project_dir, "data", "preprocessed", "plains-cree"
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


def fix_cree_punctuation(text:str):
    # Remove spaces before punctuation marks
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    # Ensure exactly one space after punctuation marks (except at the end of the sentence)
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)
    # Fix improper apostrophe spacing
    text = re.sub(r"\s+'s\b", r"'s", text)  # Attach 's to the preceding word
    text = re.sub(r"\b'\s+", r"'", text)    # Remove spaces after leading apostrophes
    # Strip any extra spaces from the start and end of the text
    return text.strip()


def clean_and_process_inuktitut_text(text:str):
    """Cleans Inuktitut text by fixing spacing between punctuation, replacing errant HTML entities, and removing other preprocessing artifacts

    Args:
        text (str): Input string to be cleaned

    Returns:
        str: cleaned string
    """
    # Fix spaces around punctuation
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)  # Remove space before punctuation
    text = re.sub(r'([.,!?;:])\s+', r'\1 ', text)  # Ensure single space after punctuation
    # Replace HTML entities and special cases
    text = unescape(text).replace("@-@", "-")
    return text.strip()


def word_count_excluding_punctuation(text:str):
    """Helper function to get word count without including punctuation

    Args:
        text (str): Input string to get word count of

    Returns:
        int: length of input string without punctuation
    """
    # Remove punctuation before counting words
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    return len(text_without_punctuation.split())


def inuktitut_process_and_filter(df: pd.DataFrame, column_to_filter="target_text", min_word_count=4):
    """Preprocess Inuktitut text data to fix punctuation spacing, remove HTML entities, and fix other noisy parts of text.
    Also filters out texts shorter than min_word_count.

    Args:
        df (pd.DataFrame): Corpus DataFrame to process text on
        column_to_filter (str, optional): Which column to filter based on word count. Defaults to "target_text".
        min_word_count (int, optional): Minimum word count for inclusion. Defaults to 4.

    Returns:
        _type_: _description_
    """
    # Apply cleaning and processing to text columns
    df["source_text"] = df["source_text"].apply(clean_and_process_inuktitut_text)
    df["target_text"] = df["target_text"].apply(clean_and_process_inuktitut_text)
    # Filter rows based on word count in the specified column
    return df[df[column_to_filter].apply(word_count_excluding_punctuation) > min_word_count]


if __name__ == '__main__':
    # Serialize all splits of Inuktitut data, for both Roman and Syllabic scripts
    for split in get_args(SPLITS):

        # Serialize and preprocess all splits for Syllabic Inuktitut
        SERIALIZED_INUKTITUT_SYLLABIC_PATH = os.path.join(
            project_dir, "data", "serialized", f"{split}_syllabic_parallel_corpus.parquet"
        )
        syllabic_parallel_corpus = load_inuktitut_parallel_corpus(INUKTITUT_SYLLABIC_PATH, split=split)
        syllabic_parallel_corpus = inuktitut_process_and_filter(syllabic_parallel_corpus)
        syllabic_parallel_corpus.to_parquet(SERIALIZED_INUKTITUT_SYLLABIC_PATH)

        # Preprocess and serialize all splits Romanized Inuktitut
        SERIALIZED_INUKTITUT_ROMAN_PATH = os.path.join(
            project_dir, "data", "serialized", f"{split}_roman_parallel_corpus.parquet"
        )
        roman_parallel_corpus = load_inuktitut_parallel_corpus(INUKTITUT_ROMAN_PATH, split=split)
        roman_parallel_corpus = inuktitut_process_and_filter(roman_parallel_corpus)
        roman_parallel_corpus.to_parquet(SERIALIZED_INUKTITUT_ROMAN_PATH)

    # Preprocess and serialize Plains Cree data
    cree_parallel_corpus = load_cree_parallel_data(CREE_PATH)
    cree_parallel_corpus = cree_parallel_corpus.map(fix_cree_punctuation)
    print(cree_parallel_corpus)
    cree_parallel_corpus.to_parquet(SERIALIZED_CREE_PATH)

    # Serialize gold standards
    serialize_gold_standards(
        input_path=GOLD_STANDARD_PATH, output_path=SERIALIZED_GOLD_STANDARD_PATH
    )