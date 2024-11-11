import os
import re
from typing import Dict, List, Set, Literal, get_args, get_origin
from sys import _getframe

import pandas as pd
from defusedxml import ElementTree as ET
from dotenv import load_dotenv
from nltk.translate.bleu_score import sentence_bleu

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
load_dotenv(dotenv_path)

SOURCE_LANGUAGES = Literal['inuktitut', 'cree']

SPLITS = Literal['test', 'train', 'dev', 'dev-dedup', 'devtest-dedup', 'devtest', 'test-dedup']

GOLD_STANDARD_MODES = Literal['consensus', 'individual']


def enforce_literals(function):
    """A helper function to enforce literals in function arguments

    Args:
        function (function object): The function for which literals are to be enforced

    Raises:
        AssertionError: Is raised if arguments are not in literal
    """
    kwargs = _getframe(1).f_locals
    for name, type_ in function.__annotations__.items():
        value = kwargs.get(name)
        options = get_args(type_)
        if get_origin(type_) is Literal and name in kwargs and value not in options:
            raise AssertionError(f"'{value}' is not in {options} for '{name}'")

def eval_results(res_df: pd.DataFrame):
    """
    Calculates the BLEU scores for each translation in the given DataFrame and adds the scores as a new column.
    Args:
        res_df (pd.DataFrame): The DataFrame containing the translation results. It should have 'tgt_txt' and 'trans_txt' columns.
    Returns:
        pd.DataFrame: The input DataFrame with an additional 'bleu_scores' column containing the BLEU scores for each translation.
    """
    bleu_scores = []
    for _, row in res_df.iterrows():
        reference = row["target_text"].split()
        prediction = row["translated_text"].split()

        bleu = sentence_bleu([reference], prediction)
        bleu_scores.append(bleu)

    res_df["bleu_scores"] = bleu_scores
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    print(f"Average BLEU score: {avg_bleu}")
    max_bleu = max(bleu_scores)
    print(f"Max BLEU score: {max_bleu}")

    return res_df

def get_file_prefixes(gs_1_path: str, gs_2_path: str) -> Set[str]:
    """
    Get unique file prefixes from the gold standard paths.

    Args:
        gs_1_path (str): The path to the first gold standard directory.
        gs_2_path (str): The path to the second gold standard directory.

    Returns:
        Set[str]: A set of unique file prefixes.
    """
    gs_1_files = os.listdir(gs_1_path)
    gs_1_prefixes = {
        os.path.join(gs_1_path, filename.split(".")[0]) for filename in gs_1_files
    }

    gs_2_files = os.listdir(gs_2_path)
    gs_2_prefixes = {
        os.path.join(gs_2_path, filename.split(".")[0]) for filename in gs_2_files
    }
    return gs_1_prefixes.union(gs_2_prefixes)

def load_parallel_text_data(
    source_directory: str,
    target_directory: str,
) -> Dict[str, List[str]]:
    """
    Load parallel text data from the source and target paths.

    Args:
        source_path (str): The path to the source text file.
        target_path (str): The path to the target text file.

    Returns:
        pd.DataFrame: The loaded parallel text data.
    """
    temp_data: Dict[str, List[str]] = {"source_text": [], "target_text": []}
    with open(source_directory, "r", encoding="utf-8") as source_file, open(
        target_directory, "r", encoding="utf-8"
    ) as target_file:
        for source_line, target_line in zip(source_file, target_file):
            source_line = source_line.strip()
            target_line = target_line.strip()
            if source_line and target_line:
                temp_data["source_text"].append(source_line)
                temp_data["target_text"].append(target_line)
    return temp_data


def load_inuktitut_parallel_corpus(path: str, split: SPLITS = 'test'):
    """Loads data from parallel corpus files specified by

    Args:
        path (str): Filepath to load without file extension

    Returns:
        pd.DataFrame: Dataframe with data from parallel corpus
    """
    enforce_literals(load_inuktitut_parallel_corpus)

    data: dict = {"source_text": [], "target_text": []}
    source_filename = f"{path}/{split}.iu"
    target_filename = f"{path}/{split}.en"

    # load data from source and target files using load_parallel_text_data
    temp_data = load_parallel_text_data(source_filename, target_filename)
    data["source_text"].extend(temp_data["source_text"])
    data["target_text"].extend(temp_data["target_text"])
    return pd.DataFrame(data)


def load_cree_parallel_data(input_directory: str) -> pd.DataFrame:
    """load Cree data from specified directory into a Dataframe

    Args:
        input_directory (str): string containing input path

    Returns:
        pd.DataFrame: Dataframe with contents from parallel data
    """
    cree_text = []
    english_text = []
    filenames = []

    for _, _, files in os.walk(input_directory):
        filenames.extend(files)

    for filename in filenames:
        if filename.endswith("_cr.txt"):
            source_path = os.path.join(input_directory, filename)
            target_path = os.path.join(input_directory, filename.replace("_cr.txt", "_en.txt"))

            if target_path:
                temp_data = load_parallel_text_data(source_path, target_path)
                cree_text.extend(temp_data["source_text"])
                english_text.extend(temp_data["target_text"])

    return pd.DataFrame({"cree_text": cree_text, "english_text": english_text})


def link_gold_standard(links, inuktitut_root, english_root):
    """
    Extracts data from the given links and returns a pandas DataFrame.

    Parameters:
    - links (list): A list of links in the alignment file.
    - inuktitut_root (ElementTree.Element): The root element of the Inuktitut XML file.
    - english_root (ElementTree.Element): The root element of the English XML file.

    Returns:
    - pd.DataFrame: A pandas DataFrame containing the extracted data.
    """
    # Create a dictionary to store the extracted data
    data = {
        "source_text": [],
        "target_text": [],
    }

    # Iterate through the links in the alignment file
    for link in links:
        # Get the xtargets and link type
        xtargets = link.get("xtargets")

        # Split the xtargets into source and target ids
        src_ids, tgt_ids = xtargets.split(";")

        # Split IDs into parts based on the link type
        src_ids = src_ids.split(" ")
        tgt_ids = tgt_ids.split(" ")
        src_phrases = [
            inuktitut_root.find(f"./p/s[@id='{sid}']").text for sid in src_ids
        ]
        tgt_phrases = [english_root.find(f"./p/s[@id='{tid}']").text for tid in tgt_ids]

        # Convert src_phrases list into one string
        src_text = " ".join(src_phrases)
        tgt_text = " ".join(tgt_phrases)

        # Add the extracted data to the data dictionary
        data["source_text"].append(src_text)
        data["target_text"].append(tgt_text)

    return pd.DataFrame(data)


def extract_and_align_gold_standard(file_prefix: str):
    """
    Extracts and aligns the text stored within the gold standards using a specified file prefix.

    Args:
        file_prefix (str): The relative file path to the gold standard, without any of the suffixes (e.g., IU-EN-Parallel-Corpus/gold-standard/annotator1-consensus/Hansard_19990401)
    """

    # Define filepaths
    # metadata_file = f"{file_prefix}.en.iu.conf" # not needed
    alignment_file = f"{file_prefix}.en.iu.xml"
    english_file = f"{file_prefix}.en.xml"
    inuktitut_file = f"{file_prefix}.iu.xml"

    # Parse the alignment file to get alignment information
    alignment_tree = ET.parse(alignment_file)
    alignment_root = alignment_tree.getroot()

    # Parse the English and Inuktitut files
    english_tree = ET.parse(english_file)
    inuktitut_tree = ET.parse(inuktitut_file)
    english_root = english_tree.getroot()
    inuktitut_root = inuktitut_tree.getroot()

    # Remove any 0-Many or Many-0 links
    zeroes = re.compile(r"0")  # regex to find zeroes in the alignment file
    links = [
        link
        for link in alignment_root.iter("link")
        if not zeroes.search(link.get("type"))
    ]

    # Return the linked gold standard as a DataFrame
    return link_gold_standard(links, inuktitut_root, english_root)


def load_consensus_gold_standards(gs_dir: str):
    """
    Load consensus gold standards from the given directory.

    Args:
        gs_dir (str): The directory path where the gold standards are located.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the concatenated gold standards.
    """
    gs_1_path = os.path.join(gs_dir, "annotator1-consensus")
    gs_2_path = os.path.join(gs_dir, "annotator2-consensus")
    print("loading consensus gold standard")

    file_prefixes = get_file_prefixes(gs_1_path, gs_2_path)

    gs_dfs = []

    for file_prefix in file_prefixes:
        df = extract_and_align_gold_standard(file_prefix)
        gs_dfs.append(df)

    return pd.concat(gs_dfs, ignore_index=True)


def load_individual_gold_standards(gs_dir: str):
    """
    Load individually annotated gold standards from the given directory.

    Args:
        gs_dir (str): The directory path where the gold standards are located.

    Returns:
        pd.DataFrame: A concatenated DataFrame of the gold standards.
    """
    gs_1_path = os.path.join(gs_dir, "annotator1")
    gs_2_path = os.path.join(gs_dir, "annotator2")
    print("loading individually annotated gold standard")

    file_prefixes = get_file_prefixes(gs_1_path, gs_2_path)

    gs_dfs = []

    for file_prefix in file_prefixes:
        df = extract_and_align_gold_standard(file_prefix)
        gs_dfs.append(df)

    return pd.concat(gs_dfs, ignore_index=True)


def serialize_gold_standards(
    input_path: str,
    output_path: str,
    mode: GOLD_STANDARD_MODES = 'consensus',
):
    """
    Loads the gold standard data for a specified mode and file prefix and saves it as a parquet file for fast access.

    Args:
        input_path (str): The directory containing the gold standard files.
        output_path (str): The path and file name for resultant parquet file.
        mode (GOLD_STANDARD_MODES): Specifies which gold standard files to load, valid types are 'consensus' and 'individual.
    """
    enforce_literals(serialize_gold_standards)

    print("Loading and Serializing Inuktitut Gold Standard")
    if os.path.exists(output_path):
        print("Serialized gold standard already exists... skipping")
        return
    if mode == 'consensus':
        gold_standard_df = load_consensus_gold_standards(input_path)
        gold_standard_df.to_parquet(output_path)
        return
    if mode == 'individual':
        gold_standard_df = load_individual_gold_standards(input_path)
        gold_standard_df.to_parquet(output_path)
        return


#TODO double check cree serializing works properly, I have it set up improperly in zero_shot.py
def serialize_parallel_corpus(
    input_path: str,
    output_path: str,
    split: SPLITS = 'test',
    language_mode: SOURCE_LANGUAGES = 'inuktitut',
):
    """
    Serializes the parallel corpus to a parquet file. Does not run if the file already exists.

    Args:
        input_path (str): Filepath to Inuktitut or Cree data to be serialized
        output_path (str): Filepath to save the serialized parallel corpus.
        split (SPLITS): Split for Inuktitut data only. Defaults to 'test'.
        mode (SOURCE_LANGUAGES): Mode for selecting language to serialize. Defaults to 'inuktitut'.
    """
    enforce_literals(serialize_parallel_corpus)
    if os.path.exists(output_path):
        print("Serialized parallel corpus already exists... skipping")
        return
    if language_mode == 'inuktitut':
        print(f"Serializing Inuktitut parallel corpus to {output_path}")
        parallel_corpus_df = load_inuktitut_parallel_corpus(input_path, split=split)
        parallel_corpus_df.to_parquet(output_path)
        return
    if language_mode == 'cree':
        print(f"Serializing Cree parallel corpus to {output_path}")
        parallel_corpus_df = load_cree_parallel_data(input_path)
        parallel_corpus_df.to_parquet(output_path)
        return


def generate_n_shot_examples(gold_standard: pd.DataFrame, n_shots: int):
    """
    Selects a random subset of examples from the gold standard and formats into a string to pass to language model.

    Args:
        gold_std (pandas.DataFrame): The gold standard dataframe.
        n_shots (int): The number of examples to include from the gold standard.

    Returns:
        str: A string containing the examples for few-shot learning.
    """
    # Select a random subset of examples from the gold standard
    gold_standard_subset = gold_standard.sample(n=n_shots, replace=False)

    # Empty string to store examples for few-shot learning
    examples = ""

    # Parse gold standard dataframe to get examples
    for _, row in gold_standard_subset.iterrows():
        source_example = row["source_text"]
        target_example = row["target_text"]

        examples += f"Text: {source_example} | Translation: {target_example} ###\n"

    return examples