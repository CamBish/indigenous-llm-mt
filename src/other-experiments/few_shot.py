#%%
import os
import time
import sys

import openai
from openai import OpenAI
import pandas as pd

from dotenv import load_dotenv

from utils import get_project_root

module_path = "/Users/cambish/code-base/indigenous-llm-mt/src"

if module_path not in sys.path:
    print("test")
    sys.path.append(module_path)

from utils import get_project_root

project_dir = get_project_root(os.path.abspath(os.path.dirname(__file__)))
dotenv_path = os.path.join(project_dir, ".env")
load_dotenv(dotenv_path)


#--------------------------------------------------
TEST_DEDUP_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "serialized",
    "test-dedup_syllabic_parallel_corpus.parquet"
)
DEV_DEDUP_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "serialized",
    "dev-dedup_syllabic_parallel_corpus.parquet"
)
DEVTEST_DEDUP_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "serialized",
    "devtest-dedup_syllabic_parallel_corpus.parquet"
)
#--------------------------------------------------
TEST_DEDUP_INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir,
    "data",
    "serialized",
    "test-dedup_roman_parallel_corpus.parquet"
)
DEV_DEDUP_INUKTITUT_SYLLABIC_PATH = os.path.join(
    project_dir,
    "data",
    "serialized",
    "dev-dedup_roman_parallel_corpus.parquet"
)
DEVTEST_DEDUP_INUKTITUT_ROMAN_PATH = os.path.join(
    project_dir,
    "data",
    "serialized",
    "devtest-dedup_roman_parallel_corpus.parquet"
)
#--------------------------------------------------
SERIALIZED_CREE_PATH = os.path.join(
    project_dir, "data", "serialized", "cree_corpus.parquet"
)
#--------------------------------------------------
SERIALIZED_GOLD_STANDARD_PATH = os.path.join(
    project_dir, "data", "serialized", "gold_standard.parquet"
)

# Load environment variables from .env file
SOURCE_LANGUAGE = os.environ.get("SOURCE_LANGUAGE", "Inuktitut (Syllabic)")
TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE", "English")
N_SHOTS = 10

MODEL = os.environ.get("MODEL", "Meta-Llama-3.1-70B-Instruct")
print("Working with:", MODEL)
print(f"Translating from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}")
print(f'{N_SHOTS}-shot experiment')

def few_shot_machine_translation(
    source_text:str,
    gold_standard:pd.DataFrame,
    n_shots:int,
    temperature=0,
    max_completion_tokens=200,
    stop=None,
    n=None,
    model=MODEL
):
    # Select a random subset of examples from the gold standard
    gold_standard_subset = gold_standard.sample(n=n_shots, replace=False)

    messages = [
        {"role": "system", "content": "You are a machine translation system."},
    ]

    source_texts = gold_standard_subset["source_text"].tolist()
    target_texts = gold_standard_subset["target_text"].tolist()

    for source, target in zip(source_texts, target_texts):
        example_prompt = [
            {
                "role": "user",
                "content": f"[{SOURCE_LANGUAGE}]: {source}",
            },
            {
                "role": "assistant",
                "content": f"[{TARGET_LANGUAGE}]: {target}",
            },
        ]
        messages.extend(example_prompt)

    messages.append({
        "role": "user",
        "content": f"[{SOURCE_LANGUAGE}]: {source_text} \n [{TARGET_LANGUAGE}]:"
    })
    
    json_data = {"model": MODEL,  "messages": messages}
    
    if temperature is not None:
        json_data["temperature"] = temperature
    if max_completion_tokens is not None:
        json_data["max_tokens"] = max_completion_tokens
    if stop is not None:
        json_data["stop"] = stop
    if n is not None:
        json_data["n"] = n
    
    print("Source text to be translated:\n", source_text)
    output = None
    while output is None:
        try:
            output = client.chat.completions.create(**json_data)
        except openai.OpenAIError as e:
            print(e)
            time.sleep(10)

    print("Generated Output:\n", output.choices[0].message.content)
    print("--------------------------------------------------")
    return output.choices[0].message.content

#%%
if __name__ == '__main__':
    try:
        openai.api_key = os.environ.get('OPENAI_API_KEY')
        if openai.api_key is None:
            raise Exception
        else:
            print("API Key Obtained Successfully!")
    except Exception:
        print("Error reading OpenAI API key from environment variable")
        exit(1)

    client = OpenAI()

    inuktitut_syllabic_df = pd.read_parquet(TEST_DEDUP_INUKTITUT_SYLLABIC_PATH)
    inuktitut_gold_standard_df = pd.read_parquet(SERIALIZED_GOLD_STANDARD_PATH)
    print("Loaded data")


    # llama-3.1-8b-instruct: 1,5,10,20
    # llama-3.1-70b-instruct: 1,5,
    print("Generating Translation Results")
    start_time = time.perf_counter()
    inuktitut_syllabic_df["response"] = inuktitut_syllabic_df['source_text'].apply(few_shot_machine_translation, args=(inuktitut_gold_standard_df,N_SHOTS))
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print("Total time elapsed:", elapsed_time)
    average_time = elapsed_time / len(inuktitut_syllabic_df.index)
    print("Average processing time:", average_time)

    out_dir = os.path.join(
        project_dir,
        "src",
        "results",
        MODEL,
        "few-shot-results"
    )

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(
        out_dir,
        f"{N_SHOTS}-few-shot.parquet"
    )

    inuktitut_syllabic_df.to_parquet(out_path)
    print('saved results to disk')
# %%