# dataset_info_tool.py
import os
import json
import pandas as pd
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

BASE_URL = "https://devapp.lungmap.net"  # Define BASE_URL for public file paths
BASE_DATASET_DIR = os.getenv('BASE_DATASET_DIR')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PLOT_OUTPUT_DIR = os.getenv('PLOT_OUTPUT_DIR')

if not BASE_DATASET_DIR:
    raise ValueError("BASE_DATASET_DIR environment variable is not set. Please set it in your .env file.")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")

# Define training data file path in standardized location
TRAIN_DATASET_INFO_FILE = os.path.join(BASE_DATASET_DIR, "training_data", "dataset_info_training_data.json")

# Initialize a lock for thread-safe file operations
log_lock = threading.Lock()

def load_log() -> dict:
    """
    Load the existing log from TRAIN_DATASET_INFO_FILE.
    If the file doesn't exist, initialize with an empty structure.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(TRAIN_DATASET_INFO_FILE), exist_ok=True)
    
    if not os.path.exists(TRAIN_DATASET_INFO_FILE):
        return {"dataset_info_queries": []}

    try:
        with open(TRAIN_DATASET_INFO_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted, reset it
        return {"dataset_info_queries": []}

def append_log(query: str, tsv_path: str = None, response: str = None) -> None:
    """
    Append a new log entry to the TRAIN_DATASET_INFO_FILE.
    
    Parameters:
    - query: The user's query.
    - tsv_path: Optional path to the TSV file analyzed.
    - response: The response provided to the user.
    """
    with log_lock:  # Ensure thread-safe access
        log_data = load_log()
        log_entry = {
            "query": query,
            "tsv_path": tsv_path,
            "response": response,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        log_data["dataset_info_queries"].append(log_entry)
        with open(TRAIN_DATASET_INFO_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

def resolve_tsv_path(tsv_path: str) -> str:
    """
    Resolve the TSV file path, handling both full paths and paths relative to BASE_DATASET_DIR.
    
    Parameters:
    - tsv_path: The provided TSV file path
    
    Returns:
    - str: The resolved absolute path
    """
    # If it's a full URL starting with BASE_URL, convert to relative path
    if tsv_path.startswith(BASE_URL):
        relative_path = "/" + tsv_path[len(BASE_URL):].lstrip("/")
    else:
        relative_path = tsv_path

    # If it's an absolute path, use it directly
    if os.path.isabs(relative_path):
        return relative_path
    
    # Check if file exists in PLOT_OUTPUT_DIR first if available
    if PLOT_OUTPUT_DIR:
        potential_path = os.path.join(PLOT_OUTPUT_DIR, os.path.basename(relative_path))
        if os.path.exists(potential_path):
            return potential_path
    
    # Otherwise, resolve relative to BASE_DATASET_DIR
    return os.path.join(BASE_DATASET_DIR, relative_path)

def general_parse_tsv_sync(tsv_path: str) -> dict:
    """
    Synchronous TSV parsing using pandas.
    """
    try:
        resolved_path = resolve_tsv_path(tsv_path)
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"TSV file not found at: {resolved_path}")
        df = pd.read_csv(resolved_path, sep='\t')
        return df.to_dict(orient='records')
    except Exception as e:
        raise ValueError(f"Error parsing TSV file: {e}")

async def general_parse_tsv(tsv_path: str) -> dict:
    """
    Asynchronously parse a TSV file by running the synchronous version in a thread pool.
    """
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, general_parse_tsv_sync, tsv_path)

async def analyze_tsv_with_llm(tsv_data: dict, query: str) -> str:
    """
    Analyze TSV data using Google's LLM.
    
    Parameters:
    - tsv_data: The parsed TSV data as a dictionary
    - query: The user's query about the TSV data
    
    Returns:
    - str: The LLM's analysis of the TSV data
    """
    # Convert TSV data to a string representation
    tsv_str = json.dumps(tsv_data, indent=2)
    
    # Construct the prompt with instructions for concise responses
    prompt = f"""Analyze the following TSV data and answer the query concisely.
    
Query: {query}

TSV Data:
{tsv_str}

"""

    # Initialize the Google LLM
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )

    # Create and send the message
    message = HumanMessage(content=prompt)
    response = model.invoke([message])
    
    return response.content.strip()

async def Dataset_Explorer(query: str = "", tsv_path: str = None) -> str:
    """
    Provides dataset information and analyzes TSV files based on the context of the query.
    """
    try:
        response = None
        # Route 2: If a TSV file path is provided, use the TSV analysis route
        if tsv_path:
            try:
                resolved_path = resolve_tsv_path(tsv_path)
                parsed_data = await general_parse_tsv(resolved_path)
                
                # Ensure query is non-empty
                if not query or query.strip() == "":
                    return "Error: A query is required when providing a TSV file for analysis. Please specify what you want to analyze about this file."
                
                # Use LLM to analyze the TSV data
                analysis = await analyze_tsv_with_llm(parsed_data, query)
                
                # Prepare structured response
                response = json.dumps({
                    "tsv_path": resolved_path,
                    "query": query,
                    "analysis": analysis,
                    "timestamp": pd.Timestamp.now().isoformat()
                }, indent=4)
                
                # Log the interaction
                append_log(query, resolved_path, response)
                
                return response
            except Exception as e:
                error_msg = f"Error analyzing TSV file '{tsv_path}': {repr(e)}"
                # Log errors too
                append_log(query, tsv_path, error_msg)
                return error_msg

        # Route 1: Default metadata parsing route.
        try:
            # Import the preloaded dataset index from the visualization tool.
            from visualization_tool import PRELOADED_DATASET_INDEX

            if PRELOADED_DATASET_INDEX is None:
                error_msg = "Error: Dataset index is not preloaded into memory."
                append_log(query, None, error_msg)
                return error_msg

            # Use the pre-parsed dataset information
            parsed_data = PRELOADED_DATASET_INDEX

            # Retrieve datasets and notes from the parsed data
            datasets = parsed_data.get("datasets", [])
            notes = parsed_data.get("notes", {})

            # Remove the directory path field from each dataset
            for dataset in datasets:
                if "**Directory Path** (File location of the dataset)" in dataset:
                    del dataset["**Directory Path** (File location of the dataset)"]

            # Prepare the structured JSON response
            response = json.dumps({"datasets": datasets, "notes": notes}, indent=4)
            
            # Log the interaction
            append_log(query, None, response)
            
            return response

        except Exception as e:
            error_msg = f"Error retrieving dataset information: {repr(e)}"
            append_log(query, None, error_msg)
            return error_msg
    except Exception as e:
        error_msg = f"Error in Dataset_Explorer: {repr(e)}"
        append_log(query, None, error_msg)
        return error_msg

# For backward compatibility
dataset_info_tool = Dataset_Explorer
