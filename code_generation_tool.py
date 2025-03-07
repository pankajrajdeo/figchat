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
    BASE_DATASET_DIR,  # <--- Rely on the imported BASE_DATASET_DIR
    PLOT_OUTPUT_DIR
)
from pathlib import Path

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
llm_base = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    max_retries=2,  # Retry up to 2 times
    timeout=30      # Prevent hanging
)  # For top-level decision, dataset selection, final output

llm_advanced = ChatOpenAI(
    model="gpt-4o",
    max_retries=2,
    timeout=30
)  # For code generation & repair

# ----------------------------
# Python REPL
# ----------------------------
python_repl_instance = PythonREPL()

# Define training data file path
TRAIN_CODE_DATA_FILE = os.path.join(BASE_DATASET_DIR, "training_data", "code_generation_training_data.json")
log_lock = threading.Lock()

# Security: Path Sanitization
def sanitize_path(path: str, base_dir: str = BASE_DATASET_DIR) -> str:
    """Sanitize file path to prevent traversal outside base directory."""
    try:
        # Handle URLs first
        BASE_URL = "https://devapp.lungmap.net"
        if isinstance(path, str) and path.startswith(BASE_URL):
            logger.info(f"Path is a URL, stripping BASE_URL: {path}")
            return path[len(BASE_URL):]
        
        # Handle local paths
        resolved_path = Path(path).resolve()
        base_path = Path(base_dir).resolve()
        if not resolved_path.is_relative_to(base_path):
            logger.error(f"Path traversal attempt detected: {path}")
            raise ValueError(f"Path outside allowed directory: {path}")
        return str(resolved_path)
    except Exception as e:
        logger.error(f"Path sanitization failed: {str(e)}")
        raise ValueError(f"Invalid path: {path}")

# Security: File Input Validation
def validate_file_input(file_input: Optional[str]) -> Optional[str]:
    """Validate file input for existence and supported types."""
    if not file_input:
        logger.info("No file_input provided")
        return None
    
    # Handle URLs, especially those starting with BASE_URL
    BASE_URL = "https://devapp.lungmap.net"
    if isinstance(file_input, str) and file_input.startswith(BASE_URL):
        logger.info(f"File input is a URL: {file_input}")
        # Extract the path part from the URL
        relative_path = file_input[len(BASE_URL):]
        # Check if it has a supported extension
        if any(relative_path.endswith(ext) for ext in ('.tsv', '.csv', '.h5ad')):
            logger.info(f"URL has supported file extension: {relative_path}")
            return file_input
        else:
            logger.warning(f"URL has unsupported file type: {file_input}")
            return None
    
    # Handle local file paths
    if not os.path.exists(file_input):
        logger.warning(f"File does not exist: {file_input}")
        return None
    if not file_input.endswith(('.tsv', '.csv', '.h5ad')):
        logger.warning(f"Unsupported file type: {file_input}")
        return None
    return file_input

def load_code_log() -> dict:
    """Load the existing log from TRAIN_CODE_DATA_FILE."""
    try:
        if not BASE_DATASET_DIR:
            logger.warning("BASE_DATASET_DIR environment variable is not set or empty")
            return {"workflow_logs": []}
        log_dir = os.path.join(BASE_DATASET_DIR, "training_data")
        os.makedirs(log_dir, exist_ok=True)
        if not os.path.exists(TRAIN_CODE_DATA_FILE):
            logger.info(f"Log file does not exist: {TRAIN_CODE_DATA_FILE}")
            return {"workflow_logs": []}
        with open(TRAIN_CODE_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.warning(f"Error decoding JSON from log file: {e}")
        return {"workflow_logs": []}
    except Exception as e:
        logger.warning(f"Error loading log file: {e}")
        return {"workflow_logs": []}

def append_code_log(entry: dict) -> None:
    """Append a new entry to the TRAIN_CODE_DATA_FILE."""
    try:
        if not BASE_DATASET_DIR:
            logger.warning("Skipping log entry as BASE_DATASET_DIR is not set")
            return
        with log_lock:
            log_data = load_code_log()
            # Ensure the 'workflow_logs' key exists so we can append
            if "workflow_logs" not in log_data:
                log_data["workflow_logs"] = []
            log_data["workflow_logs"].append(entry)
            log_dir = os.path.dirname(TRAIN_CODE_DATA_FILE)
            os.makedirs(log_dir, exist_ok=True)
            with open(TRAIN_CODE_DATA_FILE, "w", encoding="utf-8") as f:
                json.dump(log_data, f, indent=4)
    except Exception as e:
        logger.warning(f"Failed to append log entry: {e}")

# ----------------------------
# Workflow 0: Top-Level Decision
# ----------------------------
class Workflow0Model(BaseModel):
    is_file_analysis: bool

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
    [("system", WORKFLOW0_PROMPT_TEMPLATE), ("human", "{user_query}\n{file_input}")]
).partial(format_instructions=workflow0_parser.get_format_instructions())

async def run_workflow0(user_query: str, file_input: Optional[str] = None) -> Workflow0Model:
    logger.info("Workflow 0 started: Determining if query is about file analysis.")
    prompt_input = {"user_query": user_query, "file_input": file_input or "None"}
    response = await (workflow0_prompt | llm_base | workflow0_parser).ainvoke(prompt_input)
    logger.info("Workflow 0 completed: is_file_analysis=%s", response.is_file_analysis)
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

Optional File Input (if provided):
{file_input}

Your output should be a single JSON object adhering to this schema:
{format_instructions}
"""

workflow1_prompt = ChatPromptTemplate.from_messages(
    [("system", WORKFLOW1_PROMPT_TEMPLATE), ("human", "{user_query}\n{file_input}")]
).partial(format_instructions=workflow1_parser.get_format_instructions())

def get_dataset_metadata() -> str:
    global PRELOADED_DATASET_INDEX, DATASET_INDEX_FILE
    if PRELOADED_DATASET_INDEX is None:
        PRELOADED_DATASET_INDEX = parse_tsv_data(DATASET_INDEX_FILE)
    if isinstance(PRELOADED_DATASET_INDEX, pd.DataFrame):
        data_to_dump = PRELOADED_DATASET_INDEX.to_dict(orient='list')
    else:
        data_to_dump = PRELOADED_DATASET_INDEX
    return json.dumps(data_to_dump, indent=4)

async def run_workflow1(user_query: str, file_input: Optional[str] = None) -> Workflow1Model:
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
    logger.debug("Filtered metadata for dataset '%s' => %d items found.", dataset_name, len(filtered["datasets"]))
    return filtered

def extract_code(generated_text: str) -> str:
    if "<think>" in generated_text and "</think>" in generated_text:
        generated_text = generated_text.split("</think>")[-1].strip()
    match = re.search(r"```(?:python)?\s*(.*?)```", generated_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return generated_text.strip()

def filter_unsafe_code(code: str) -> str:
    """Filter out unsafe operations from generated code."""
    unsafe_patterns = [
        r'os\.remove', r'os\.system', r'sys\.exit', r'__import__\(\s*[\'"]os[\'"]\)', 
        r'__import__\(\s*[\'"]sys[\'"]\)', r'shutil\.rmtree'
    ]
    for pattern in unsafe_patterns:
        if re.search(pattern, code):
            logger.error(f"Unsafe code detected and blocked: {pattern}")
            raise ValueError("Generated code contains unsafe operations")
    return code

def repair_and_rerun_code(original_code: str, error_message: str, dataset_metadata: Optional[str] = None, file_input: Optional[str] = None) -> tuple[str, str]:
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
                   "Make sure to find appropriate field names from the provided metadata for the user query be it plot, analysis, etc."
                   "Output ONLY the code, ensure it ends with a print statement referencing final file paths or results. "
                   "For security: Do not generate code that manipulates or deletes internal files, modifies system resources, "
                   "or abuses system privileges (e.g., no os.remove, os.system, sys.exit, or similar commands)."),
        ("human", repair_prompt)
    ]

    repaired_msg = llm_advanced.invoke(ChatPromptTemplate.from_messages(prompt_messages).format_messages())
    repaired_code = extract_code(repaired_msg.content)
    repaired_code = filter_unsafe_code(repaired_code)
    logger.info("Repaired code:\n%s", repaired_code)
    new_output = python_repl_instance.run(repaired_code)
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
    logger.info("Workflow 2/3: Generating code with dataset_name='%s', file_input='%s'", dataset_name, file_input)
    try:
        if file_input is not None:
            if not isinstance(file_input, str):
                logger.warning(f"file_input is not a string: {type(file_input).__name__}")
                file_input = str(file_input)
            logger.info(f"Using validated file_input: {file_input}")
        
        if dataset_metadata:
            filtered_meta = filter_metadata(json.loads(dataset_metadata), dataset_name)
            meta_str_escaped = json.dumps(filtered_meta, indent=2).replace("{", "{{").replace("}", "}}")
        else:
            meta_str_escaped = "None"

        # Prepare a clear message about the file input for the LLM
        file_input_message = "None"
        if file_input:
            file_input_message = f"{file_input} (This is a local file path, not a URL. The BASE_URL has already been stripped if it was present.)"

        code_prompt = f"""
# --- Metadata and Preloaded Files ---
# Optional Dataset Metadata: {meta_str_escaped}
# Preloaded .h5ad files: {list(PRELOADED_DATA.keys())}
# Base dataset directory: {BASE_DATASET_DIR}
# Plot output directory: {PLOT_OUTPUT_DIR}
# Optional File Input (if provided): {file_input_message}

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
"""

        system_msg = (
            "You are a scanpy code generation assistant that generates Python code using scanpy and other important libraries for scRNA-seq analysis."
            "Use pandas for TSV/CSV files or scanpy for .h5ad files as appropriate. "
            "Ensure the code processes the file input if provided, or uses dataset metadata if available, "
            "Make sure to find appropriate field names from the provided metadata for the user query be it plot, analysis, etc."
            "saves plots to PLOT_OUTPUT_DIR, and ends with a print statement referencing final file paths or results. "
            "Output ONLY the code. "
            "For security: Do not generate code that manipulates or deletes internal files, modifies system resources, "
            "or abuses system privileges (e.g., no os.remove, os.system, sys.exit, or similar commands)."
        )

        prompt_messages = ChatPromptTemplate.from_messages(
            [("system", system_msg), ("human", code_prompt)]
        ).format_messages()

        logger.info("Generating code with LLM...")
        code_response = llm_advanced.invoke(prompt_messages)
        logger.info("Code generation complete")
        generated_code = extract_code(code_response.content)
        generated_code = filter_unsafe_code(generated_code)
        logger.info("Generated code:\n%s", generated_code)

        error_markers = ("Traceback", "Error", "Exception")
        successful_code = generated_code
        failed_code = None

        try:
            logger.info("Executing generated code...")
            output = python_repl_instance.run(generated_code)
            logger.info("Code execution complete")
        except BaseException as e:
            logger.error("Error during initial code execution: %s", str(e))
            try:
                output, successful_code = repair_and_rerun_code(generated_code, str(e), dataset_metadata, file_input)
            except Exception as repair_error:
                logger.error("Error during code repair: %s", str(repair_error))
                output = f"Initial execution error: {str(e)}\nRepair error: {str(repair_error)}"
                successful_code = None
                failed_code = generated_code
        else:
            if any(m in output for m in error_markers):
                logger.warning("Error in initial code output. Attempting repair.")
                try:
                    output, successful_code = repair_and_rerun_code(generated_code, output, dataset_metadata, file_input)
                except Exception as repair_error:
                    logger.error("Error during code repair: %s", str(repair_error))
                    output = f"Initial output contained errors: {output[:100]}...\nRepair error: {str(repair_error)}"
                    successful_code = None
                    failed_code = generated_code
        
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
    except Exception as e:
        error_message = f"Unhandled error in generate_code_and_execute: {str(e)}"
        logger.error(error_message)
        return error_message, None, None

# ----------------------------
# Final Output Workflow
# ----------------------------
class FinalOutputModel(BaseModel):
    content: Optional[str] = None
    generated_code: Optional[str] = None
    error: Optional[str] = None
    selected_dataset: Optional[str] = None
    selected_datasets: Optional[List[str]] = None
    dataset_path_placeholder: Optional[str] = None
    output_directory_placeholder: Optional[str] = None

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
    [("system", FINAL_OUTPUT_PROMPT_TEMPLATE), ("human", "{raw_output}\n{generated_code}\n{selected_datasets}")]
).partial(format_instructions=final_output_parser.get_format_instructions())

# ----------------------------
# Top-Level Orchestrator with Modularization
# ----------------------------
async def determine_query_type(user_query: str, file_input: str) -> Workflow0Model:
    intermediate_json = create_status_json("Determining query type...")
    logger.info("Returning intermediate status: %s", intermediate_json)
    return await run_workflow0(user_query, file_input)

async def process_file_analysis(user_query: str, file_input: str) -> tuple:
    intermediate_json = create_status_json("File analysis detected, generating code...")
    logger.info("Returning intermediate status: %s", intermediate_json)
    logger.info("File analysis detected, skipping dataset selection.")
    return generate_code_and_execute(user_query, file_input=file_input)

async def process_dataset_analysis(user_query: str, file_input: str) -> tuple:
    intermediate_json = create_status_json("Dataset analysis detected, selecting dataset...")
    logger.info("Returning intermediate status: %s", intermediate_json)
    logger.info("Dataset analysis detected, proceeding with dataset selection.")
    workflow1_result = await run_workflow1(user_query, file_input)
    dataset_name = workflow1_result.dataset_name or "ALL"
    dataset_metadata = get_dataset_metadata()
    return generate_code_and_execute(user_query, dataset_name, file_input, dataset_metadata), dataset_name

def create_status_json(message: str, status: str = "in_progress"):
    return json.dumps({
        "content": message,
        "code": "",
        "datasets": "",
        "status": status,
        "timestamp": pd.Timestamp.now().isoformat()
    })

async def Code_Generator(user_query: str, file_input: Optional[str] = None) -> str:
    """
    Generates and executes custom Python code based on user queries, with optional file input processing.
    
    Parameters:
    - user_query: A string containing the user's code generation request
    - file_input: Optional path to a file to analyze (TSV, CSV, H5AD, PDF, PNG, etc.)
      * Can be a URL starting with https://devapp.lungmap.net/ (will be automatically converted to a local path)
      * Can be a local path to a file
      * Particularly useful for analyzing TSV files generated by the visualization_tool
    
    Returns:
    - A JSON string containing:
      - content: Human-readable output of the code execution
      - code: The generated Python code
      - datasets: Datasets used in the analysis
      - error: Any error messages (if applicable)
      
    Notes:
    - When passed a file_input, the function will analyze its contents based on the user_query
    - TSV files from visualization_tool often contain network data, gene lists, or statistical results
      that can be further analyzed with custom code
    - The function will automatically strip the BASE_URL from file inputs to access the local file
    """
    logger.info("Top-level orchestrator started for user query:\n%s, file_input: %s", user_query, file_input)
    try:
        BASE_URL = "https://devapp.lungmap.net"
        stripped_file_input = None
        if file_input:
            try:
                validated_input = validate_file_input(file_input)
                if validated_input:
                    stripped_file_input = sanitize_path(validated_input)
                    logger.info(f"Using sanitized file_input: {stripped_file_input}")
                else:
                    logger.info(f"File input validation failed, proceeding with None")
            except ValueError as e:
                logger.error(f"Path sanitization error: {str(e)}")
                stripped_file_input = None
            except Exception as e:
                logger.error(f"Error processing file_input: {e}")
                stripped_file_input = None
        
        workflow0_result = await determine_query_type(user_query, stripped_file_input)
        logger.info("After workflow0, continuing execution with is_file_analysis=%s", workflow0_result.is_file_analysis)
        
        if workflow0_result.is_file_analysis:
            raw_output, successful_code, failed_code = await process_file_analysis(user_query, stripped_file_input)
            selected_datasets = None
        else:
            (raw_output, successful_code, failed_code), dataset_name = await process_dataset_analysis(user_query, stripped_file_input)
            datasets = [
                "HLCA_full_superadata_v3_norm_log_deg.h5ad",
                "HCA_fetal_lung_normalized_log_deg.h5ad",
                "BPD_infant_Sun_normalized_log_deg.h5ad",
                "BPD_fetal_normalized_log_deg.h5ad"
            ]
            selected_datasets = datasets if dataset_name.upper() == "ALL" else [dataset_name]
        
        logger.info("BEFORE final output generation")
        prompt_input = {
            "raw_output": raw_output or "Error occurred during execution",
            "generated_code": successful_code or failed_code or "",
            "selected_datasets": ", ".join(selected_datasets) if selected_datasets else "None"
        }
        try:
            final_output_response = await (final_output_prompt | llm_base | final_output_parser).ainvoke(prompt_input)
            logger.info("AFTER final output generation - succeeded")
        except Exception as e:
            logger.error("ERROR in final output generation: %s", str(e))
            from pydantic import BaseModel
            class SimpleResponse(BaseModel):
                content: str
                code: str
                datasets: str
                error: Optional[str]
            final_output_response = SimpleResponse(
                content="Error occurred during processing",
                code=successful_code or failed_code or "",
                datasets=", ".join(selected_datasets) if selected_datasets else "None",
                error=str(e)
            )
            logger.info("AFTER final output generation - handled exception")
        
        log_entry = {
            "workflow": "Final Output Generation",
            "raw_output": raw_output,
            "generated_code": successful_code or failed_code,
            "selected_datasets": selected_datasets,
            "final_output": final_output_response.dict(),
            "original_file_input": file_input,
            "stripped_file_input": stripped_file_input,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        append_code_log(log_entry)
        
        try:
            output_dict = final_output_response.dict()
            output_dict["status"] = "complete"
            final_output_json = json.dumps(output_dict)
        except Exception as e:
            logger.error("ERROR serializing output to JSON: %s", str(e))
            final_output_json = json.dumps({
                "content": "Error occurred during execution",
                "code": successful_code or failed_code or "",
                "datasets": ", ".join(selected_datasets) if selected_datasets else "None",
                "error": str(e),
                "status": "complete"
            })
        
        logger.info("Top-level orchestrator complete. Output as JSON:\n%s", final_output_json)
        return final_output_json
    except Exception as e:
        logger.error("Unhandled exception in Code_Generator: %s", str(e))
        error_json = json.dumps({
            "content": f"An error occurred: {str(e)}",
            "code": "",
            "datasets": "",
            "error": str(e),
            "status": "complete"
        })
        logger.info("Returning error JSON: %s", error_json)
        return error_json

# For backward compatibility
code_generation_tool = Code_Generator
