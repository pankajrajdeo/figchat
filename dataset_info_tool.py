# dataset_info_tool.py
import json
from utils import parse_tsv_data  # Import the shared parsing utility

def dataset_info_tool() -> str:
    """
    Provides structured and detailed metadata information about all the h5ad datasets we are working with,
    excluding the directory path from the output.
    """
    from visualization_tool import PRELOADED_DATASET_INDEX  # Use the preloaded dataset index

    if PRELOADED_DATASET_INDEX is None:
        return "Error: Dataset index is not preloaded into memory."

    try:
        # Use the pre-parsed dataset information
        parsed_data = PRELOADED_DATASET_INDEX

        # Retrieve datasets and notes from the parsed data
        datasets = parsed_data.get("datasets", [])
        notes = parsed_data.get("notes", {})

        # Remove the `**Directory Path**` field from each dataset
        for dataset in datasets:
            if "**Directory Path** (File location of the dataset)" in dataset:
                del dataset["**Directory Path** (File location of the dataset)"]

        # Return the structured JSON as a string
        return json.dumps({"datasets": datasets, "notes": notes}, indent=4)

    except Exception as e:
        return f"Error retrieving dataset information: {repr(e)}"


