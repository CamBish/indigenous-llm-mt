#%%
import os
import re
import pandas as pd

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
        r"\[English\]:\s*(.*?)(?:\n|$)",             # Capture text after [English]:
        r"Translation:\s*\"(.*?)\"",                # Capture text in quotes after Translation:
        r"translates to:\s*\"(.*?)\"",              # Capture text in quotes after translates to:
        r"translates to:\s*(.*?)(?:\n|$)"           # Capture text after translates to: without quotes
    ]

    translations = []

    for pattern in patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        translations.extend(matches)  # Collect all matches

    # Clean up the translations (e.g., strip extra whitespace)
    translations = [t.strip() for t in translations]
    
    return translations

#%%
if __name__ == "__main__":
    
    examples = [
        '''
        [Inuktitut]: ᐅᖃᖅᑎ ( ᐊᖓᔪᖅᑳᖅ ᔪᐊᔾ ᖁᓚᐅᑦ ): ᑐᒃᓯᐊᕐᓂᒃᑯᑦ ᐅᒃᑯᐃᕈᓐᓇᖅᐱᐅᒃ, ᒥᔅᑐ ᕿᓐᖑᖅ.
        [English]: Uqaqti (Angajuqaq Juaj Qallut): Tukisianiikuttuq ukuaqirunnanqapiuk, mista qinnguq.

        Translation: Uqaqti (Angajuqaq Juaj Qallut
        ''',
        '''
        (Inuktitut text) 

        Here is the translation:

        "Hello, in Nunavut, the Inuit are the majority, and the Inuktitut language is the most widely spoken language, and it is the working language of the government, and it is also the language of instruction in schools."
        ''',
        '''
        (Inuktitut text provided)

        Here is the translation:

        [English]: The Inuit have traditionally been fishers and hunters, and still hunt and fish to this day, but many Inuit have also become involved in the tourism and service industries.
        ''',
        '''
        ᐅᖃᖅᑏ, ᐅᑯᐊ ᖁᕕᐊᓲᑎᒋᔪᒪᕙᒃᑲ:

        Translation: "Hello, how are you today?"
        ''',
        '''
        I'm happy to help with the translation!

        ᖁᔭᓕᑦᑎᐊᖅᐳᖓ ᐊᒃᓱᐊᓗᒃ translates to:


        "I'm happy to see you."

        Here's a breakdown of the translation:

        * ᖁᔭᓕᑦᑎᐊᖅᐳᖓ (kuujjautiqaqpuqunga) means "I'm happy" or "I'm glad"
        * ᐊᒃᓱᐊᓗᒃ (akusualuk) means "to see you" or "to see someone"

        Note: Inuktitut is a polysynthetic language, which means that words are composed of many morphemes that convey different meanings. The translation may not be word-for-word, but rather
        '''
    ]
    


    for i, example in enumerate(examples, 1):
        print(f"Example {i}:")
        translations = extract_translation(example)
        for t in translations:
            print(f"  - {t}")
        print()

#%%
    # Define the input directory
    input_directory = "/Users/cambish/code-base/indigenous-llm-mt/src/results"
    
    # Find all Parquet files in the input directory
    parquet_files = find_parquet_files(input_directory)
    
    # Convert each Parquet file to Excel
    for parquet_file in parquet_files:
        convert_parquet_to_excel(parquet_file)