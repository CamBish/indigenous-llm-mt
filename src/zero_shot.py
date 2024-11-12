import os
# import re
# import time
import argparse

import dotenv
# import pandas as pd
# import pyarrow.parquet as pq
# from IPython.display import display

from llama3_inference import TransformersWrapper

# from utils import eval_results

project_dir = os.path.join(os.path.dirname(__file__), os.pardir)
dotenv_path = os.path.join(project_dir, ".env")
dotenv.load_dotenv(dotenv_path)

# define constants
TARGET_LANGUAGE="English"

#TODO redo for remote environment
TEST_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "split",
    "test",
)

TEST_INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir,
    "data",
    "external",
    "Nunavut-Hansard-Inuktitut-English-Parallel-Corpus-3.0",
    "split",
    "test",
)

TEST_SERIALIZED_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir, "data", "serialized", "test_syllabic_parallel_corpus.parquet"
)
TEST_SERIALIZED_INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir, "data", "serialized", "test_roman_parallel_corpus.parquet"
)
SERIALIZED_GOLD_STANDARD_PATH = os.path.join(
    project_dir, "data", "serialized", "gold_standard.parquet"
)
SERIALIZED_CREE_PATH = os.path.join(
    project_dir, "data", "serialized", "cree_corpus.parquet"
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Define Parameters")
    parser.add_argument('source_language', type=str, default='Inuktitut', choices=['inuktitut', 'cree'] , help='Language to perform experiments on')
    parser.add_argument('-m', '--model_path', type=str, default="~/projects/def-zhu2048/cambish/llama3_1_8b_instruct", help="Location of model files")
    parser.add_argument('-n','--n_samples', type=int, default=1000, help="Number of samples to test from dataset")
    parser.add_argument('-t',"--temp", type=float, default=0.0, help = "temperature for sampling")
    parser.add_argument('-mx',"--max_len", type=int, default=200, help = "max number of tokens in answer")
    args = parser.parse_args()
    
    #TODO parse arguments - especially data file paths
    # split model path and use last directory as name
    MODEL_NAME = args.model_path.split('/')[-1]
    
    SOURCE_LANGUAGE = args.source_language
    
    #TODO Load data using utils
    
    
    #TODO create prompts
    
    
    model = TransformersWrapper(model_path=args.model_path)
    
    #TODO model call + post processing