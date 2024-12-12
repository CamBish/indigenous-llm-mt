#%%
import os
import time

import openai
from openai import OpenAI
import pandas as pd

from dotenv import load_dotenv

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
TRANSLITERATION_LANGUAGE = os.environ.get("TRANSLITERATION_LANGUAGE", "Inuktitut (Romanized)")
TARGET_LANGUAGE = os.environ.get("TARGET_LANGUAGE", "English")

MODEL = os.environ.get("MODEL", "Meta-Llama-3.1-8B-Instruct")
print("Working with:", MODEL)
print(f"Translating from {SOURCE_LANGUAGE} to {TARGET_LANGUAGE}")

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
    sys_prompt = f"""You are a machine translation system that operates in two steps.
    
    The user will provide the source text marked by [{SOURCE_LANGUAGE}]. 
    Transliterate the text into Roman characters with the prefix [{TRANSLITERATION_LANGUAGE}]:

    Then translate the romanized text from before with the prefix [{TARGET_LANGUAGE}]:
    """
    # Select a random subset of examples from the gold standard
    gold_standard_subset = gold_standard.sample(n=n_shots, replace=False)
    
    messages = [
        {"role": "system", "content": sys_prompt},
    ]

    source_texts = gold_standard_subset["source_text"].tolist()
    target_texts = gold_standard_subset["target_text"].tolist()

    messages.extend([
        [
            {"role": "user", "content": f"[{SOURCE_LANGUAGE}]: {src}"},
            {"role": "assistant", "content": f"[Inuktitut (Romanized)]: \n[{TARGET_LANGUAGE}]: {tgt}"}
        ]
        for src, tgt in zip(source_texts, target_texts)
    ])

    
    translation_prompt = {
        "role": "user",
        "content": f"[{SOURCE_LANGUAGE}]: {source_text} \n [{TARGET_LANGUAGE}]:"
    }
    messages.append(translation_prompt)
    
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
    inuktitut_romanized_df = pd.read_parquet(TEST_DEDUP_INUKTITUT_ROMAN_PATH)
    inuktitut_gold_standard_df = pd.read_parquet(SERIALIZED_GOLD_STANDARD_PATH)

    inuktitut_gold_standard_df.rename(columns={
    "src_lang": "source_text",
    "tgt_lang": "target_text"
    }, inplace=True)
    print(inuktitut_gold_standard_df.columns.tolist())


    inuktitut_gold_standard_df.to_parquet(SERIALIZED_GOLD_STANDARD_PATH)


    print("Loaded data")

    print(inuktitut_gold_standard_df)

    print("Generating Translation Results")
    start_time = time.perf_counter()
    inuktitut_syllabic_df["response"] = inuktitut_syllabic_df['source_text'].apply(few_shot_machine_translation, args=(inuktitut_gold_standard_df,5))
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print("Total time elapsed:", elapsed_time)
    average_time = elapsed_time / len(inuktitut_syllabic_df.index)
    print("Average processing time:", average_time)
    
    romanized_pattern = r"Romanization: (.+)\n"
    translated_pattern = r"Translation: (.+)(\n|$)"
    
    inuktitut_syllabic_df["romanized_text"] = inuktitut_syllabic_df["response"].str.extract(romanized_pattern)
    inuktitut_syllabic_df["translated_text"] = inuktitut_syllabic_df["response"].str.extract(translated_pattern)
    
    inuktitut_syllabic_df["romanized_truth"] = inuktitut_romanized_df["source_text"]
    
    out_dir = os.path.join(
        project_dir,
        "src",
        "results",
        MODEL
    )
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    out_path = os.path.join(
        out_dir,
        "syllabic-zero-shot.parquet"
    )
    
    inuktitut_syllabic_df.to_parquet(out_path)
# %%