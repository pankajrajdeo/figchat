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
llm_base = ChatOpenAI(model="gpt-4o-mini-2024-07-18")  # For top-level decision, dataset selection, and final output
llm_advanced = ChatOpenAI(model="gpt-4o")             # For code generation & repair

# ----------------------------
# Python REPL
# ----------------------------
python_repl_instance = PythonREPL()

# Define training data file path
BASE_DATASET_DIR = os.environ.get("BASE_DATASET_DIR", "")
TRAIN_CODE_DATA_FILE = os.path.join(BASE_DATASET_DIR, "training_data", "code_generation_training_data.json")
log_lock = threading.Lock()

def load_code_log() -> dict:
    """
    Load the existing log from TRAIN_CODE_DATA_FILE.
    """
    os.makedirs(os.path.dirname(TRAIN_CODE_DATA_FILE), exist_ok=True)
    if not os.path.exists(TRAIN_CODE_DATA_FILE):
        return {"workflow_logs": []}
    try:
        with open(TRAIN_CODE_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"workflow_logs": []}

def append_code_log(entry: dict) -> None:
    """
    Append a new entry to the TRAIN_CODE_DATA_FILE.
    """
    with log_lock:
        log_data = load_code_log()
        log_data["workflow_logs"].append(entry)
        with open(TRAIN_CODE_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

# ----------------------------
# Workflow 0: Top-Level Decision
# ----------------------------

class Workflow0Model(BaseModel):
    is_file_analysis: bool  # True if query is about input file analysis and file_input is provided

workflow0_parser = PydanticOutputParser(pydantic_object=Workflow0Model)

WORKFLOW0_PROMPT_TEMPLATE = """\
Determine whether the user query is specifically about analyzing an input file and whether a file input is provided.

User Query:
{user_query}

File Input:
{file_input}

Instructions:
- If the query explicitly requests analysis of an input file (e.g., "load this file", "find columns in this file") and a file input is provided (not "None"), set 'is_file_analysis' to **true**.
- Otherwise, set 'is_file_analysis' to **false**.

Output a single JSON object adhering to this schema:
{format_instructions}
"""

workflow0_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WORKFLOW0_PROMPT_TEMPLATE),
        ("human", "{user_query}\n{file_input}")
    ]
).partial(format_instructions=workflow0_parser.get_format_instructions())

async def run_workflow0(user_query: str, file_input: Optional[str] = None) -> Workflow0Model:
    """
    Workflow 0: Decide if the query is about input file analysis and a file is provided.
    """
    logger.info("Workflow 0 started: Determining if query is about file analysis.")
    prompt_input = {
        "user_query": user_query,
        "file_input": file_input or "None"
    }

    response = await (workflow0_prompt | llm_base | workflow0_parser).ainvoke(prompt_input)
    logger.info("Workflow 0 completed: is_file_analysis=%s", response.is_file_analysis)
    
    # Log Workflow 0 training data
    log_entry = {
        "workflow": "Workflow 0",
        "user_query": user_query,
        "file_input": file_input,
        "output": response.dict(),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    append_code_log(log_entry)
    
    return response

# ----------------------------
# Workflow 1: Dataset Selection (Optional)
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

Optional File Input (if provided):
{file_input}

Your output should be a single JSON object adhering to this schema:
{format_instructions}
"""

workflow1_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", WORKFLOW1_PROMPT_TEMPLATE),
        ("human", "{user_query}\n{file_input}")
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

async def run_workflow1(user_query: str, file_input: Optional[str] = None) -> Workflow1Model:
    """
    Workflow 1: Select the most relevant dataset based on user query, optional file input, and metadata.
    """
    logger.info("Workflow 1 started: Selecting dataset based on user query.")
    dataset_metadata_str = get_dataset_metadata()
    
    prompt_input = {
        "user_query": user_query,
        "file_input": file_input or "None",
        "dataset_metadata": dataset_metadata_str
    }

    chain_messages = workflow1_prompt.format_messages(**prompt_input)
    logger.debug("Workflow 1 prompt messages: %s", chain_messages)

    response = await (workflow1_prompt | llm_base | workflow1_parser).ainvoke(prompt_input)
    logger.info("Workflow 1 completed: dataset_name=%s", response.dataset_name)
    
    # Log Workflow 1 training data
    log_entry = {
        "workflow": "Workflow 1",
        "user_query": user_query,
        "file_input": file_input,
        "dataset_metadata": dataset_metadata_str,
        "output": response.dict(),
        "timestamp": pd.Timestamp.now().isoformat()
    }
    append_code_log(log_entry)
    
    return response

# ----------------------------
# Workflow 2 & 3: Code Generation & Repair
# ----------------------------

def filter_metadata(metadata: dict, selection: Union[str, None]) -> dict:
    """
    Filter metadata for a specific dataset or return all if selection is 'ALL'.
    """
    if not selection or selection.upper() == "ALL":
        logger.debug("No dataset specified or 'ALL', returning full metadata.")
        return metadata

    dataset_name = selection
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
    Extract Python code from generated text, removing <think> blocks and capturing code in triple backticks.
    """
    if "<think>" in generated_text and "</think>" in generated_text:
        generated_text = generated_text.split("</think>")[-1].strip()

    match = re.search(r"```(?:python)?\s*(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()

def repair_and_rerun_code(original_code: str, error_message: str, dataset_metadata: Optional[str] = None, file_input: Optional[str] = None) -> tuple[str, str]:
    """
    Repair failed code and re-execute it, with optional dataset metadata.
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
# Optional Dataset Metadata: {dataset_metadata or "None"}
# Optional File Input: {file_input or "None"}
-----
Provide a corrected version of the code that:
1. Ensures output directories exist (os.makedirs(..., exist_ok=True)).
2. Fixes any path or argument mistakes.
3. Ends with a print statement listing final file path(s) or results.
Output ONLY the corrected Python code.
"""

    prompt_messages = [
        ("system", "You are a scanpy code generation assistant that generates Python code using scanpy and other important libraries for scRNA-seq analysis."
                 "Output ONLY the code, ensure it ends with a print statement referencing final file paths or results."),
        ("human", repair_prompt)
    ]

    repaired_msg = llm_advanced.invoke(ChatPromptTemplate.from_messages(prompt_messages).format_messages())
    repaired_code = extract_code(repaired_msg.content)
    logger.info("Repaired code:\n%s", repaired_code)

    new_output = python_repl_instance.run(repaired_code)
    
    # Log repair attempt
    log_entry = {
        "workflow": "Code Repair",
        "original_code": original_code,
        "error_message": error_message,
        "repaired_code": repaired_code,
        "output": new_output,
        "dataset_metadata": dataset_metadata,
        "file_input": file_input,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    append_code_log(log_entry)
    
    return new_output, repaired_code

def generate_code_and_execute(user_query: str, dataset_name: Optional[str] = None, file_input: Optional[str] = None, dataset_metadata: Optional[str] = None) -> tuple[str, Optional[str], Optional[str]]:
    """
    Generate and execute code based on user query, optional dataset, and optional file input.
    Dataset metadata is optional.
    Returns (output, successful_code, failed_code).
    """
    logger.info("Workflow 2/3: Generating code with dataset_name='%s'", dataset_name)
    
    # Use dataset metadata only if provided
    if dataset_metadata:
        filtered_meta = filter_metadata(json.loads(dataset_metadata), dataset_name)
        meta_str_escaped = json.dumps(filtered_meta, indent=2).replace("{", "{{").replace("}", "}}")
    else:
        meta_str_escaped = "None"

    code_prompt = f"""
# --- Metadata and Preloaded Files ---
# Optional Dataset Metadata: {meta_str_escaped}
# Preloaded .h5ad files: {list(PRELOADED_DATA.keys())}
# Base dataset directory: {BASE_DATASET_DIR}
# Plot output directory: {PLOT_OUTPUT_DIR}
# Optional File Input (if provided): {file_input or "None"}

# --- User Request ---
# User Query: {user_query}

# --- Task ---
# Generate Python code that:
# 1. If a file input is provided, loads and processes it as per the query (e.g., using pandas for TSV files).
# 2. If no file input is provided and dataset metadata is available, loads the specified dataset (e.g., using scanpy for .h5ad files).
# 3. Performs the analysis as described in the User Query.
# 4. If not specified, generates a .txt, .tsv, or appropriate file based on the user request and shows output in the terminal.
# 5. Saves any generated plots into the PLOT_OUTPUT_DIR.
# 6. Ends with a print statement listing file paths or results.

# Output ONLY the code and nothing else.
"""

    system_msg = (
        "You are a scanpy code generation assistant that generates Python code using scanpy and other important libraries for scRNA-seq analysis."
        "Use pandas for TSV/CSV files or scanpy for .h5ad files as appropriate. "
        "Ensure the code processes the file input if provided, or uses dataset metadata if available, "
        "saves plots to PLOT_OUTPUT_DIR, and ends with a print statement referencing final file paths or results. "
        "Output ONLY the code."
    )

    prompt_messages = ChatPromptTemplate.from_messages(
        [("system", system_msg), ("human", code_prompt)]
    ).format_messages()

    code_response = llm_advanced.invoke(prompt_messages)
    generated_code = extract_code(code_response.content)
    logger.info("Generated code:\n%s", generated_code)

    error_markers = ("Traceback", "Error", "Exception")
    successful_code = generated_code
    failed_code = None

    try:
        output = python_repl_instance.run(generated_code)
    except BaseException as e:
        logger.error("Error during initial code execution: %s", str(e))
        output, successful_code = repair_and_rerun_code(generated_code, str(e), dataset_metadata, file_input)
    else:
        if any(m in output for m in error_markers):
            logger.warning("Error in initial code output. Attempting repair.")
            output, successful_code = repair_and_rerun_code(generated_code, output, dataset_metadata, file_input)
    
    # Log code generation and execution
    log_entry = {
        "workflow": "Code Generation and Execution",
        "user_query": user_query,
        "dataset_name": dataset_name,
        "file_input": file_input,
        "dataset_metadata": dataset_metadata,
        "generated_code": generated_code,
        "output": output,
        "successful_code": successful_code,
        "failed_code": failed_code if not successful_code else None,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    append_code_log(log_entry)

    if any(m in output for m in error_markers):
        logger.error("Repaired code also failed: %s", output)
        failed_code = successful_code
        successful_code = None

    logger.info("Workflow 2/3 complete, final output:\n%s", output)
    return output, successful_code, failed_code

# ----------------------------
# Final Output Workflow
# ----------------------------

class FinalOutputModel(BaseModel):
    content: Optional[str] = None           # Processed output message (e.g., with BASE_URL prepended)
    generated_code: Optional[str] = None    # Code with placeholders for paths
    error: Optional[str] = None             # Error message if execution fails
    selected_dataset: Optional[str] = None  # Single dataset name (if applicable)
    selected_datasets: Optional[List[str]] = None  # List of datasets (for "ALL" case)
    dataset_path_placeholder: Optional[str] = None  # Placeholder for dataset paths
    output_directory_placeholder: Optional[str] = None  # Placeholder for output directories

final_output_parser = PydanticOutputParser(pydantic_object=FinalOutputModel)

FINAL_OUTPUT_PROMPT_TEMPLATE = """\
Based on the provided raw output, generated code, and selected dataset(s), generate a final JSON output adhering to the following schema:

{format_instructions}

Raw Output:
{raw_output}

Generated Code:
{generated_code}

Selected Dataset(s):
{selected_datasets}

Instructions:
- For the 'content' field, use the full raw output and prepend 'https://devapp.lungmap.net' to any file paths (e.g., paths starting with '/' or containing '.tsv', '.png', etc.) within it, preserving all other text.
**IMPORTANT**
 - ALWAYS PREPEND 'https://devapp.lungmap.net' TO THE FILE PATHS IN THE RAW OUTPUT.
 - MAKE SURE TO REPLACE THE DATASET PATHS IN THE GENERATED CODE WITH '/path_to/<dataset_name>' for each dataset used (e.g., '/path_to/BPD_fetal_normalized_log_deg.h5ad') if a dataset is used.
 - MAKE SURE TO REPLACE THE FILE INPUT PATHS IN THE GENERATED CODE WITH '/path_to/<filename>' (e.g., '/path_to/venn_overlapping_markers_42693.tsv') if a file input is used.
 - MAKE SURE TO REPLACE THE OUTPUT DIRECTORIES IN THE GENERATED CODE WITH 'path_to_output_directory'.
- If the raw output contains an error (e.g., 'Traceback', 'Error', 'Exception'), include it in the 'error' field and set 'content' to 'Error occurred during execution'.
- Include the selected dataset(s) in 'selected_dataset' (for a single dataset) or 'selected_datasets' (for multiple) if datasets are used; otherwise, leave as null if only a file input is used.
- Populate 'dataset_path_placeholder' with the dataset path used (e.g., '/path_to/BPD_fetal_normalized_log_deg.h5ad') if a dataset is used, or the file input path (e.g., '/path_to/venn_overlapping_markers_42693.tsv') if a file input is used, or leave as null if multiple datasets are used.
- Set 'output_directory_placeholder' to 'path_to_output_directory'.

Output a single JSON object matching the schema.
"""

final_output_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", FINAL_OUTPUT_PROMPT_TEMPLATE),
        ("human", "{raw_output}\n{generated_code}\n{selected_datasets}")
    ]
).partial(format_instructions=final_output_parser.get_format_instructions())

# ----------------------------
# Top-Level Orchestrator
# ----------------------------

async def Code_Generator(user_query: str, file_input: Optional[str] = None) -> str:
    """
    Orchestrate workflows based on whether the query is about file analysis or dataset analysis.
    Returns a JSON string with the processed output.
    """
    logger.info("Top-level orchestrator started for user query:\n%s, file_input: %s", user_query, file_input)
    
    # Define BASE_URL and strip it from file_input once at the top
    BASE_URL = "https://devapp.lungmap.net"
    stripped_file_input = file_input[len(BASE_URL):] if file_input and file_input.startswith(BASE_URL) else file_input
    
    # Step 0: Determine if this is a file analysis query
    workflow0_result = await run_workflow0(user_query, stripped_file_input)
    
    if workflow0_result.is_file_analysis:
        # File analysis path: Skip Workflow 1, go directly to Workflow 2
        logger.info("File analysis detected, skipping dataset selection.")
        raw_output, successful_code, failed_code = generate_code_and_execute(user_query, file_input=stripped_file_input)
        selected_datasets = None  # No datasets selected for file analysis
    else:
        # Dataset analysis path: Proceed with Workflow 1, then Workflow 2
        logger.info("Dataset analysis detected, proceeding with dataset selection.")
        workflow1_result = await run_workflow1(user_query, stripped_file_input)
        dataset_name = workflow1_result.dataset_name or "ALL"
        dataset_metadata = get_dataset_metadata()
        
        datasets = [
            "HLCA_full_superadata_v3_norm_log_deg.h5ad",
            "HCA_fetal_lung_normalized_log_deg.h5ad",
            "BPD_infant_Sun_normalized_log_deg.h5ad",
            "BPD_fetal_normalized_log_deg.h5ad"
        ]
        selected_datasets = datasets if dataset_name.upper() == "ALL" else [dataset_name]
        
        raw_output, successful_code, failed_code = generate_code_and_execute(
            user_query, dataset_name, stripped_file_input, dataset_metadata
        )
    
    # Step 3: Generate final output using LLM workflow
    prompt_input = {
        "raw_output": raw_output or "Error occurred during execution",
        "generated_code": successful_code or failed_code or "",
        "selected_datasets": ", ".join(selected_datasets) if selected_datasets else "None"
    }
    
    final_output_response = await (final_output_prompt | llm_base | final_output_parser).ainvoke(prompt_input)
    
    # Log final output generation
    log_entry = {
        "workflow": "Final Output Generation",
        "raw_output": raw_output,
        "generated_code": successful_code or failed_code,
        "selected_datasets": selected_datasets,
        "final_output": final_output_response.dict(),
        "original_file_input": file_input,  # Log original file_input for reference
        "stripped_file_input": stripped_file_input,
        "timestamp": pd.Timestamp.now().isoformat()
    }
    append_code_log(log_entry)
    
    # Convert Pydantic model to JSON string
    final_output_json = final_output_response.json()
    
    logger.info("Top-level orchestrator complete. Output as JSON:\n%s", final_output_json)
    return final_output_json

# For backward compatibility
code_generation_tool = Code_Generator
