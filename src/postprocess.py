import os
import pandas as pd

def find_parquet_files(directory):
    parquet_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    return parquet_files

#TODO Update so it doesn't rewrite files if they already exist
# Function to convert Parquet to Excel
def convert_parquet_to_excel(parquet_file):
    # Read the Parquet file
    df = pd.read_parquet(parquet_file)
    
    # Create output file path (same folder as the Parquet file, with .xlsx extension)
    base_name = os.path.splitext(os.path.basename(parquet_file))[0]
    output_file = os.path.join(os.path.dirname(parquet_file), f"{base_name}.xlsx")
    
    # Save the DataFrame to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Converted {parquet_file} to {output_file}")

if __name__ == "__main__":
    # Define the input directory
    input_directory = "/Users/cambish/code-base/indigenous-llm-mt/src/results"
    
    # Find all Parquet files in the input directory
    parquet_files = find_parquet_files(input_directory)
    
    # Convert each Parquet file to Excel
    for parquet_file in parquet_files:
        convert_parquet_to_excel(parquet_file)