#%%
import os
import time

import openai
from openai import OpenAI
import pandas as pd

from dotenv import load_dotenv
# from IPython.display import display

from utils import get_project_root

project_dir = get_project_root(os.path.abspath(os.path.dirname(__file__)))
dotenv_path = os.path.join(project_dir, ".env")
load_dotenv(dotenv_path)

#--------------------------------------------------
SERIALIZED_CREE_PATH = os.path.join(
    project_dir, "data", "serialized", "cree_corpus.parquet"
)
#--------------------------------------------------
SOURCE_LANGUAGE = "Plains Cree"
TARGET_LANGUAGE = "English"

MODEL = os.environ.get("MODEL", "Meta-Llama-3.1-8B-Instruct")
print("Working with:", MODEL)

def zero_shot_machine_translation(
    source_text:str,
    temperature=0,
    max_completion_tokens=350,
    stop=None,
    n=None,
    model=None,
):
    messages = [
        {
            "role": "system",
            "content": "You are a machine translation system.",
        },
        {
            "role": "user",
            "content": f"[{SOURCE_LANGUAGE}]: {source_text}\n[{TARGET_LANGUAGE}]:",
        },
    ]

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

    plains_cree_df = pd.read_parquet(SERIALIZED_CREE_PATH)
    print("Loaded data")

    print("Generating Translation Results")
    start_time = time.perf_counter()
    plains_cree_df["response"] = plains_cree_df["source_text"].apply(zero_shot_machine_translation)
    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print("Total time elapsed:", elapsed_time)
    average_time = elapsed_time / len(plains_cree_df.index)
    print("Average processing time:", average_time)
    
    
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
        "plains-cree-zero-shot.parquet"
    )
    
    plains_cree_df.to_parquet(out_path)
    print("Saved results to disk")
# %%