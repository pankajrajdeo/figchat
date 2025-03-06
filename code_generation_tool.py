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
llm_base = ChatOpenAI(model="gpt-4o-mini-2024-07-18")  # For dataset selection and final output
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
    Workflow 1: Select the most relevant dataset based on user query and metadata.
    """
    logger.info("Workflow 1 started: Selecting dataset based on user query.")
    dataset_metadata_str = get_dataset_metadata()
    prompt_input = {
        "user_query": user_query,
        "dataset_metadata": dataset_metadata_str
    }

    chain_messages = workflow1_prompt.format_messages(**prompt_input)
    logger.debug("Workflow 1 prompt messages: %s", chain_messages)

    response = await (workflow1_prompt | llm_base | workflow1_parser).ainvoke(prompt_input)
    logger.info("Workflow 1 completed: dataset_name=%s", response.dataset_name)
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

def repair_and_rerun_code(original_code: str, error_message: str) -> tuple[str, str]:
    """
    Repair failed code and re-execute it, ensuring directories exist and ending with a print statement.
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
        ("system", "You are a scanpy code generation assistant that corrects Python code using scanpy for scRNA-seq data analysis. "
                 "Output ONLY the code, ensure it ends with a print statement referencing final file paths or results."),
        ("human", repair_prompt)
    ]

    repaired_msg = llm_advanced.invoke(ChatPromptTemplate.from_messages(prompt_messages).format_messages())
    repaired_code = extract_code(repaired_msg.content)
    logger.info("Repaired code:\n%s", repaired_code)

    new_output = python_repl_instance.run(repaired_code)
    return new_output, repaired_code

def generate_code_and_execute(user_query: str, dataset_name: str) -> tuple[str, Optional[str], Optional[str]]:
    """
    Generate and execute code based on user query and dataset, repairing if necessary.
    Returns (output, successful_code, failed_code).
    """
    logger.info("Workflow 2/3: Generating code for dataset='%s'", dataset_name)
    global PRELOADED_DATASET_INDEX

    if not PRELOADED_DATASET_INDEX:
        logger.error("No preloaded dataset index found.")
        return "No preloaded dataset index is available.", None, None

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
        "You are a scanpy code generation assistant that generates Python code using scanpy for scRNA-seq data analysis. "
        "Ensure the code loads all preloaded .h5ad files if 'ALL' is selected, processes data per the user request, "
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
        output, successful_code = repair_and_rerun_code(generated_code, str(e))
        append_code_log(user_query, generated_code, output, str(e))
    else:
        append_code_log(user_query, generated_code, output)
        if any(m in output for m in error_markers):
            logger.warning("Error in initial code output. Attempting repair.")
            output, successful_code = repair_and_rerun_code(generated_code, output)
            append_code_log(user_query, generated_code, output, "Repair attempted")

    if any(m in output for m in error_markers):
        logger.error("Repaired code also failed: %s", output)
        failed_code = successful_code
        successful_code = None

    logger.info("Workflow 2/3 complete, final output:\n%s", output)
    return output, successful_code, failed_code

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
        return {"code_generations": []}
    try:
        with open(TRAIN_CODE_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {"code_generations": []}

def append_code_log(user_query: str, generated_code: str, output: str, error: Optional[str] = None) -> None:
    """
    Append a new code generation entry to the TRAIN_CODE_DATA_FILE.
    """
    with log_lock:
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
- Replace dataset paths in the generated code with '/path_to/<dataset_name>' for each dataset used (e.g., '/path_to/BPD_fetal_normalized_log_deg.h5ad').
- Replace output directories in the generated code with 'path_to_output_directory'.
- Prepend 'https://devapp.lungmap.net' to file paths in the raw output to form the 'content' field.
- If the raw output contains an error (e.g., 'Traceback', 'Error', 'Exception'), include it in the 'error' field and set 'content' to 'Error occurred during execution'.
- Include the selected dataset(s) in 'selected_dataset' (for a single dataset) or 'selected_datasets' (for multiple).
- Populate 'dataset_path_placeholder' with the dataset path used (e.g., '/path_to/BPD_fetal_normalized_log_deg.h5ad') or leave as null if multiple datasets are used.
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

async def Code_Generator(user_query: str) -> str:
    """
    Orchestrate dataset selection, code generation, execution, and final output formatting using an LLM workflow.
    Returns a JSON string with the processed output.
    """
    logger.info("Top-level orchestrator started for user query:\n%s", user_query)
    
    # Step 1: Dataset selection
    workflow1_result = await run_workflow1(user_query)
    dataset_name = workflow1_result.dataset_name or "ALL"
    
    # Step 2: Code generation and execution
    raw_output, successful_code, failed_code = generate_code_and_execute(user_query, dataset_name)
    
    # Determine selected datasets
    datasets = [
        "HLCA_full_superadata_v3_norm_log_deg.h5ad",
        "HCA_fetal_lung_normalized_log_deg.h5ad",
        "BPD_infant_Sun_normalized_log_deg.h5ad",
        "BPD_fetal_normalized_log_deg.h5ad"
    ]
    selected_datasets = datasets if dataset_name.upper() == "ALL" else [dataset_name]
    
    # Prepare input for the final output LLM
    prompt_input = {
        "raw_output": raw_output or "Error occurred during execution",
        "generated_code": successful_code or failed_code or "",
        "selected_datasets": ", ".join(selected_datasets)
    }
    
    # Step 3: Generate final output using LLM workflow
    final_output_response = await (final_output_prompt | llm_base | final_output_parser).ainvoke(prompt_input)
    
    # Convert Pydantic model to JSON string
    final_output_json = final_output_response.json()
    
    logger.info("Top-level orchestrator complete. Output as JSON:\n%s", final_output_json)
    return final_output_json

# For backward compatibility
code_generation_tool = Code_Generator
