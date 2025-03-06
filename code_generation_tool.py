# code_generation_tool.py
import os
import json
import re
import logging
import pandas as pd
from typing import Union, List, Optional, Literal
import asyncio
from pydantic import BaseModel 
import scanpy as sc 
import threading
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from utils import parse_tsv_data
from preload_datasets import (
    PRELOADED_DATA,
    PRELOADED_DATASET_INDEX,
    DATASET_INDEX_FILE,
    BASE_DATASET_DIR,
    PLOT_OUTPUT_DIR
)

# ----------------------------
# Configure Logging
# ----------------------------
logger = logging.getLogger("code_generation_tool")
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# ----------------------------
# LLM Models
# ----------------------------
llm_base = ChatOpenAI(model="gpt-4o-mini-2024-07-18")  # For dataset selection
llm_advanced = ChatOpenAI(model="gpt-4o")             # For code generation & repair

# ----------------------------
# Python REPL
# ----------------------------
python_repl_instance = PythonREPL()

# ----------------------------
# Workflow 1: Dataset Selection
# ----------------------------

class Workflow1Model(BaseModel):
    dataset_name: Optional[Literal[
        "HLCA_full_superadata_v3_norm_log_deg.h5ad",
        "HCA_fetal_lung_normalized_log_deg.h5ad",
        "BPD_infant_Sun_normalized_log_deg.h5ad",
        "BPD_fetal_normalized_log_deg.h5ad",
        "ALL"
    ]]

workflow1_parser = PydanticOutputParser(pydantic_object=Workflow1Model)

WORKFLOW1_PROMPT_TEMPLATE = """\
Based on the user's query and the available dataset metadata, perform the following tasks:

**Dataset Selection:**
   - Select the most relevant dataset from the available options.

Dataset Metadata:
{dataset_metadata}

User Query:
{user_query}

Your output should be a single JSON object adhering to this schema:
{format_instructions}
"""

workflow1_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WORKFLOW1_PROMPT_TEMPLATE),
        ("human", "{user_query}")
    ]
).partial(format_instructions=workflow1_parser.get_format_instructions())


def get_dataset_metadata() -> str:
    """
    Return the entire dataset metadata as a JSON string, using parse_tsv_data if needed.
    """
    global PRELOADED_DATASET_INDEX, DATASET_INDEX_FILE
    if PRELOADED_DATASET_INDEX is None:
        PRELOADED_DATASET_INDEX = parse_tsv_data(DATASET_INDEX_FILE)
    if isinstance(PRELOADED_DATASET_INDEX, pd.DataFrame):
        data_to_dump = PRELOADED_DATASET_INDEX.to_dict(orient='list')
    else:
        data_to_dump = PRELOADED_DATASET_INDEX
    return json.dumps(data_to_dump, indent=4)


async def run_workflow1(user_query: str) -> Workflow1Model:
    """
    Workflow 1:
      - Provide entire dataset metadata to the LLM.
      - The LLM selects the most relevant dataset (or ALL) and outputs JSON adhering to Workflow1Model.
    """
    logger.info("Workflow 1 started: Selecting dataset based on user query.")
    dataset_metadata_str = get_dataset_metadata()
    prompt_input = {
        "user_query": user_query,
        "dataset_metadata": dataset_metadata_str
    }

    # Format the prompt messages
    chain_messages = workflow1_prompt.format_messages(**prompt_input)
    # Log them for debugging
    logger.debug("Workflow 1 prompt messages: %s", chain_messages)

    # Invoke the LLM with the chain messages
    response = await (workflow1_prompt | llm_base | workflow1_parser).ainvoke(prompt_input)

    logger.info("Workflow 1 completed: dataset_name=%s", response.dataset_name)
    return response


# ----------------------------
# Workflow 2 & 3: Code Generation & Repair
# ----------------------------

def filter_metadata(metadata: dict, selection: Union[str, None]) -> dict:
    """
    Return either the full metadata if selection=ALL, or filter to the single dataset name.
    Adjust if user needs multiple.
    """
    if not selection or selection.upper() == "ALL":
        logger.debug("No dataset specified or 'ALL', returning full metadata.")
        return metadata

    dataset_name = selection  # If a single name
    filtered = {
        "datasets": [
            d for d in metadata.get("datasets", [])
            if d.get("**Dataset Metadata**", {}).get("**Dataset Name**") == dataset_name
        ],
        "notes": metadata.get("notes", {})
    }
    logger.debug("Filtered metadata for dataset '%s' => %d items found.",
                 dataset_name, len(filtered["datasets"]))
    return filtered


def extract_code(generated_text: str) -> str:
    """
    Extract only the Python code from the generated text.
    Removes any <think> block, then uses regex to capture code in triple backticks.
    """
    # Remove <think> blocks
    if "<think>" in generated_text and "</think>" in generated_text:
        generated_text = generated_text.split("</think>")[-1].strip()

    # Regex to find code fenced by ```
    match = re.search(r"```(?:python)?\s*(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return generated_text.strip()


def repair_and_rerun_code(original_code: str, error_message: str) -> str:
    """
    If code fails, ask llm_advanced to fix it, ensuring:
      - Directories exist
      - Any path or argument errors are resolved
      - Must end with a print statement referencing final file path(s) or result
      - Output only code
    Then re-execute in the Python REPL.
    """
    logger.warning("Code repair workflow triggered due to error: %s", error_message)
    repair_prompt = f"""
The following Python code resulted in an error:
-----
{original_code}
-----
Error message:
{error_message}
-----
Additional info:
# Preloaded .h5ad files: {list(PRELOADED_DATA.keys())}
# Base dataset directory: {BASE_DATASET_DIR}
# Plot output directory: {PLOT_OUTPUT_DIR}
-----
Provide a corrected version of the code that:
1. Ensures output directories exist (os.makedirs(..., exist_ok=True)).
2. Fixes any path or argument mistakes.
3. Ends with a print statement listing final file path(s) or results.
Output ONLY the corrected Python code.
"""

    prompt_messages = [
        (
            "system",
            "You are a helpful assistant that corrects Python code using scanpy. "
            "Output ONLY the code, ensure it ends with a print statement referencing final file paths or results."
        ),
        ("human", repair_prompt)
    ]

    # Invoke repair LLM
    logger.debug("Repair prompt messages: %s", prompt_messages)
    repaired_msg = llm_advanced.invoke(ChatPromptTemplate.from_messages(prompt_messages).format_messages())
    repaired_code = extract_code(repaired_msg.content)
    logger.info("Repaired code:\n%s", repaired_code)

    # Rerun the repaired code
    new_output = python_repl_instance.run(repaired_code)
    return new_output


# Define training data file path in standardized location
BASE_DATASET_DIR = os.environ.get("BASE_DATASET_DIR", "")
TRAIN_CODE_DATA_FILE = os.path.join(BASE_DATASET_DIR, "training_data", "code_generation_training_data.json")

# Initialize a lock for thread-safe file operations
log_lock = threading.Lock()

def load_code_log() -> dict:
    """
    Load the existing log from TRAIN_CODE_DATA_FILE.
    If the file doesn't exist, initialize with an empty structure.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(TRAIN_CODE_DATA_FILE), exist_ok=True)
    
    if not os.path.exists(TRAIN_CODE_DATA_FILE):
        return {"code_generations": []}

    try:
        with open(TRAIN_CODE_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted, reset it
        return {"code_generations": []}


def append_code_log(user_query: str, generated_code: str, output: str, error: Optional[str] = None) -> None:
    """
    Append a new code generation entry to the TRAIN_CODE_DATA_FILE.
    
    Parameters:
    - user_query: The user query that prompted code generation
    - generated_code: The code generated by the LLM
    - output: The output or result of the code execution
    - error: Any error encountered during code execution
    """
    with log_lock:  # Ensure thread-safe access
        log_data = load_code_log()
        log_entry = {
            "user_query": user_query,
            "generated_code": generated_code,
            "output": output,
            "error": error,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        log_data["code_generations"].append(log_entry)
        with open(TRAIN_CODE_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)


def generate_code_and_execute(user_query: str, dataset_name: str) -> str:
    """
    Workflow to:
      - Filter metadata for dataset_name (or all).
      - Generate code that ends with print statement referencing file path(s)/results.
      - Execute code, if error => run repair.
    """
    logger.info("Workflow 2/3: Generating code for dataset='%s'", dataset_name)
    # Use the preloaded global metadata
    global PRELOADED_DATASET_INDEX

    if not PRELOADED_DATASET_INDEX:
        logger.error("No preloaded dataset index found.")
        return "No preloaded dataset index is available."

    # Filter metadata
    filtered_meta = filter_metadata(PRELOADED_DATASET_INDEX, dataset_name)
    meta_str_escaped = json.dumps(filtered_meta, indent=2).replace("{", "{{").replace("}", "}}")

    code_prompt = f"""
# --- Metadata and Preloaded Files ---
# Available Dataset Metadata:
{meta_str_escaped}

# Preloaded .h5ad files: {list(PRELOADED_DATA.keys())}
# Base dataset directory: {BASE_DATASET_DIR}
# Plot output directory: {PLOT_OUTPUT_DIR}

# --- User Request ---
# User Query: {user_query}

# --- Task ---
# Generate Python code using scanpy that:
# 1. Loads all appropriate .h5ad files if 'ALL' is selected, otherwise load the specified dataset.
# 2. Performs the analysis as described in the User Query.
# 3. If not specified, generate a .txt or .tsv or appropriate file based on the user request along with showing the output in the terminal.
# 4. Saves any generated plots into the PLOT_OUTPUT_DIR.
# 5. Ends with a print statement listing file paths or results.

# Output ONLY the code and nothing else.
"""

    system_msg = (
        "You are a helpful assistant that generates Python code using scanpy. "
        "Ensure the code loads all preloaded .h5ad files if 'ALL' is selected, processes data per the user request, "
        "saves plots to PLOT_OUTPUT_DIR, and ends with a print statement referencing final file paths or results. "
        "Output ONLY the code."
    )

    prompt_messages = ChatPromptTemplate.from_messages(
        [
            ("system", system_msg),
            ("human", code_prompt)
        ]
    ).format_messages()

    # Invoke advanced LLM for code generation
    code_response = llm_advanced.invoke(prompt_messages)
    generated_code = extract_code(code_response.content)
    logger.info("Generated code:\n%s", generated_code)

    # Run the code
    try:
        output = python_repl_instance.run(generated_code)
    except BaseException as e:
        logger.error("Error during code execution: %s", str(e))
        output = repair_and_rerun_code(generated_code, str(e))
        append_code_log(user_query, generated_code, output, str(e))
    else:
        append_code_log(user_query, generated_code, output)

    # If there's an error, attempt repair
    error_markers = ("Traceback", "Error", "Exception")
    if any(m in output for m in error_markers):
        logger.warning("Error in generated code. Attempting repair.")
        output = repair_and_rerun_code(generated_code, output)
        append_code_log(user_query, generated_code, output, "Repair attempted")

    logger.info("Workflow 2/3 complete, final output:\n%s", output)
    return output

# ----------------------------
# Top-Level Orchestrator
# ----------------------------
async def Code_Generator(user_query: str) -> str:
    """
    - Step 1: Workflow 1 => choose dataset (or ALL) based on user query + full metadata.
    - Step 2: Use selected dataset in code generation & run in Python REPL.
    - Step 3: If error => repair & rerun automatically.
    - Return final output or error message.
    """
    # 1) Dataset selection
    logger.info("Top-level orchestrator started for user query:\n%s", user_query)
    workflow1_result = await run_workflow1(user_query)
    dataset_name = workflow1_result.dataset_name

    if not dataset_name:
        logger.info("No dataset selected, defaulting to ALL.")
        dataset_name = "ALL"

    # 2) Code generation + execution
    final_output = generate_code_and_execute(user_query, dataset_name)

    # Modify the final output to be a JSON object
    if isinstance(final_output, str):
        # If the output is a string (content), wrap it in a JSON object
        final_output_json = json.dumps({"content": final_output})
    elif isinstance(final_output, list):
        # If the output is a list of files, wrap it in a JSON object
        final_output_json = json.dumps({"files": final_output})
    else:
        # Default case, just convert to JSON
        final_output_json = json.dumps({"output": final_output})

    logger.info("Top-level orchestrator complete. Output as JSON:\n%s", final_output_json)
    return final_output_json


# For backward compatibility
code_generation_tool = Code_Generator
    
