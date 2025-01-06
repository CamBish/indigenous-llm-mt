#%%
import os
import re
import pandas as pd

from sacrebleu.metrics import BLEU, CHRF

def find_parquet_files(directory):
    """Finds a list of all nested parquet files in a given directory

    Args:
        directory (path-like object): Directory to search for parquet files in

    Returns:
        List[path]: A list with all the paths to parquet files
    """
    parquet_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

def convert_parquet_to_excel(parquet_file):
    """Function to convert a parquet file into an xlsx file for easy human-modification

    Args:
        parquet_file (path-like object): path to parquet file
    """
    
    # Create output file path (same folder as the Parquet file, with .xlsx extension)
    base_name = os.path.splitext(os.path.basename(parquet_file))[0]
    output_file = os.path.join(os.path.dirname(parquet_file), f"{base_name}.xlsx")
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        print(f"File {output_file} already exists. Skipping conversion.")
        return

    # Read the Parquet file
    df = pd.read_parquet(parquet_file)

    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Converted {parquet_file} to {output_file}")

def extract_translation(text):
    patterns = [
        r"\[English\]:\s*(.*?)(?:\n|$)",            # Capture text after [English]:
        r"Translation:\s*\"(.*?)\"",                # Capture text in quotes after Translation:
        r"translates to:\s*(.*?)(?:\n|$)",          # Capture text after translates to: without quotes
        r"translation:\s*(.*?)(?:\n|$)",            # Capture text after translation:
    ]

    translations = []

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        translations.extend(matches)  # Collect all matches

    # Clean up the translations (e.g., strip extra whitespace and remove [English]: prefix)
    translations = [t.strip(' "').removeprefix("[English]: ") for t in translations]
    # remove duplicate strings
    translations = [t for t in set(translations)]

    return translations

def clean_results(text:str):
    # remove common prefixes in predicted translation
    return text.removeprefix("(Inuktitut): ").removeprefix("[Inuktitut]: ").removeprefix("[English]: ")


def calculate_sentence_bleu(hypothesis_text: str, target_text: str):
    """When applied to a Pandas DataFrame, this calculates the sentence-level BLEU score for each row

    Args:
        hypothesis_text (str): translation hypothesis to be tested
        target_text (str): ground truth to compare hypothesis to

    Returns:
        float: BLEU score for the sentence
    """
    bleu = BLEU(effective_order=True)
    sentence_bleu = bleu.sentence_score(hypothesis_text, [target_text])
    return sentence_bleu.score

def calculate_sentence_chrf(hypothesis_text: str, target_text: str):
    """When applied to a Pandas DataFrame, this calculates the sentence-level CHRF score for each row

    Args:
        hypothesis_text (str): translation hypothesis to be tested
        target_text (str): ground truth to compare hypothesis to

    Returns:
        float: CHRF score for the sentence
    """
    chrf = CHRF()
    sentence_chrf = chrf.sentence_score(hypothesis_text, [target_text])
    return sentence_chrf.score


#%%
if __name__ == "__main__":
    dataframe_path = '/Users/cambish/code-base/indigenous-llm-mt/src/results/Meta-Llama-3.1-70B-Instruct/syllabic-zero-shot.parquet'

    df = pd.read_parquet(dataframe_path)
    df["hypothesis_text"] = df["response"].apply(clean_results)
    # calculate sentence-level BLEU
    df["sentence_bleu"] = df["hypothesis_text"].apply(
        lambda row: calculate_sentence_bleu(row["hypothesis_text"], row["target_text"])
    )
    # calculate sentence-level CHRF
    df["sentence_chrf"] = df["hypothesis_text"].apply(
        lambda row: calculate_sentence_chrf(row["hypothesis_text"], row["target_text"])
    )
    # get target and hypothesis text as list for corpus-level evaluation
    references = df["target_text"].to_list()
    hypotheses = df["hypothesis_text"].to_list()

    bleu = BLEU()
    corpus_bleu = bleu.corpus_score(hypotheses, references)
    
    chrf = CHRF()
    corpus_chrf = chrf.corpus_score(hypotheses, references)

#%%
    # Define the input directory
    input_directory = "/Users/cambish/code-base/indigenous-llm-mt/src/results"
    
    # Find all Parquet files in the input directory
    parquet_files = find_parquet_files(input_directory)
    
    # Convert each Parquet file to Excel
    for parquet_file in parquet_files:
        df = pd.read_parquet(parquet_file)
        convert_parquet_to_excel(parquet_file)
# %%