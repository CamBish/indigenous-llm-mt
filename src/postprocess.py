import os
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

if __name__ == "__main__":
    # Define the input directory
    input_directory = "/Users/cambish/code-base/indigenous-llm-mt/src/results"
    
    # Find all Parquet files in the input directory
    parquet_files = find_parquet_files(input_directory)
    
    # Convert each Parquet file to Excel
    for parquet_file in parquet_files:
        convert_parquet_to_excel(parquet_file)