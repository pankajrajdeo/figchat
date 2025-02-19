# dataset_info_tool.py
import os
import json
import pandas as pd

BASE_URL = "https://devapp.lungmap.net"  # Define BASE_URL for public file paths

def general_parse_tsv(tsv_path: str) -> dict:
    """
    A general TSV parsing function that uses pandas to parse a TSV file.
    
    Parameters:
    - tsv_path (str): Path to the TSV file.
    
    Returns:
    - dict: The TSV content converted to a dictionary (list of records).
    """
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        return df.to_dict(orient='records')
    except Exception as e:
        raise ValueError(f"Error parsing TSV file: {e}")

def dataset_info_tool(query: str = "", tsv_path: str = None) -> str:
    """
    Provides dataset information based on the context of the query.

    Routes:
    1. Metadata Parsing Route (default):
       - Uses the preloaded dataset metadata (from visualization_tool.PRELOADED_DATASET_INDEX).
       - Strips out the directory path field before returning the JSON string.
    2. General TSV Parsing Route:
       - If a tsv_path is provided, parse that TSV file using a general TSV parsing function
         and return its parsed output as JSON.
         If the tsv_path begins with BASE_URL, remove that portion and use the relative path.

    Parameters:
    - query: A string that may influence route decisions (reserved for future context-based routing).
    - tsv_path: If provided, the TSV file path to parse and return its contents.
    """
    # Route 2: If a TSV file path is provided, use the general TSV parsing route.
    if tsv_path:
        # Remove BASE_URL if present, ensuring the path has a leading slash.
        if tsv_path.startswith(BASE_URL):
            relative_tsv_path = "/" + tsv_path[len(BASE_URL):].lstrip("/")
        else:
            relative_tsv_path = tsv_path

        try:
            parsed_data = general_parse_tsv(relative_tsv_path)
            return json.dumps(parsed_data, indent=4)
        except Exception as e:
            return f"Error parsing TSV file '{relative_tsv_path}': {repr(e)}"

    # Route 1: Default metadata parsing route.
    try:
        # Import the preloaded dataset index from the visualization tool.
        from visualization_tool import PRELOADED_DATASET_INDEX

        if PRELOADED_DATASET_INDEX is None:
            return "Error: Dataset index is not preloaded into memory."

        # Use the pre-parsed dataset information
        parsed_data = PRELOADED_DATASET_INDEX

        # Retrieve datasets and notes from the parsed data
        datasets = parsed_data.get("datasets", [])
        notes = parsed_data.get("notes", {})

        # Remove the directory path field from each dataset
        for dataset in datasets:
            if "**Directory Path** (File location of the dataset)" in dataset:
                del dataset["**Directory Path** (File location of the dataset)"]

        # Return the structured JSON as a string
        return json.dumps({"datasets": datasets, "notes": notes}, indent=4)

    except Exception as e:
        return f"Error retrieving dataset information: {repr(e)}"
