import os
import json
import pandas as pd
import asyncio
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

def resolve_tsv_path(tsv_path: str) -> str:
    """
    Resolve the TSV file path, handling both full paths and paths relative to BASE_DATASET_DIR.
    Also handles network interaction TSV files from the plot output directory.
    
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
    
    # Check if it's a network interactions file
    if relative_path.endswith('network_interactions.tsv'):
        if PLOT_OUTPUT_DIR:
            return os.path.join(PLOT_OUTPUT_DIR, relative_path)
        else:
            raise ValueError("PLOT_OUTPUT_DIR environment variable is not set. Please set it in your .env file.")
    
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

IMPORTANT INSTRUCTIONS:
1. Avoid unnecessary repetition or verbose explanations
2. Focus on the specific aspects asked in the query
3. Do NOT repeat the query or restate what the data represents"""

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
    Provides dataset information based on the context of the query.

    Routes:
    1. Metadata Parsing Route (default):
       - Uses the preloaded dataset metadata (from visualization_tool.PRELOADED_DATASET_INDEX).
       - Strips out the directory path field before returning the JSON string.
    2. TSV Analysis Route:
       - If a tsv_path is provided, parses the TSV file and uses Google's LLM to analyze it based on the query.
       - The path can be:
         - A full URL starting with BASE_URL
         - An absolute path
         - A relative path (will be resolved against BASE_DATASET_DIR)

    Parameters:
    - query: A string that may influence route decisions or specific questions about the TSV data.
    - tsv_path: If provided, the TSV file path to parse and analyze.
    """
    # Route 2: If a TSV file path is provided, use the TSV analysis route
    if tsv_path:
        try:
            resolved_path = resolve_tsv_path(tsv_path)
            parsed_data = await general_parse_tsv(resolved_path)
            
            # Use LLM to analyze the TSV data
            analysis = await analyze_tsv_with_llm(parsed_data, query)
            
            # Return structured response
            return json.dumps({
                "tsv_path": resolved_path,
                "query": query,
                "analysis": analysis,
                "timestamp": pd.Timestamp.now().isoformat()
            }, indent=4)
        except Exception as e:
            return f"Error analyzing TSV file '{tsv_path}': {repr(e)}"

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

# For backward compatibility
dataset_info_tool = Dataset_Explorer
