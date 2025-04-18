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

log_lock = threading.Lock()

# Security: Path Sanitization
def sanitize_path(path: str, base_dir: str = BASE_DATASET_DIR) -> str:
    """Sanitize and validate a file path."""
    try:
        # Convert to absolute path if relative
        if not os.path.isabs(path):
            path = os.path.join(base_dir, path)
        
        # Normalize the path
        path = os.path.normpath(path)
        
        # Ensure the path is within the base directory
        if not path.startswith(base_dir):
            raise ValueError(f"Path {path} is outside the base directory {base_dir}")
        
        return path
    except Exception as e:
        logger.error(f"Error sanitizing path {path}: {e}")
        raise

def validate_file_input(file_input: Optional[str]) -> Optional[str]:
    """Validate and sanitize file input path."""
    if not file_input:
        return None
    
    try:
        # Basic validation
        if not isinstance(file_input, str):
            raise ValueError(f"file_input must be a string, got {type(file_input)}")
        
        # Remove any BASE_URL prefix if present
        if file_input.startswith("BASE_URL/"):
            file_input = file_input[9:]  # Remove "BASE_URL/" prefix
        
        # Sanitize the path
        return sanitize_path(file_input)
    except Exception as e:
        logger.error(f"Error validating file input {file_input}: {e}")
        raise

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
        "BPD_Sun_normalized_log_deg.h5ad",
        "BPD_Sucre_normalized_log_deg.h5ad",
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

FINAL_OUTPUT_PROMPT_TEMPLATE = f"""\
Generate a structured output from the code execution results according to these guidelines:

- For the 'content' field, use the full raw output and prepend '{os.environ["BASE_URL"]}' to any file paths (e.g., paths starting with '/' or containing '.tsv', '.png', etc.) within it, preserving all other text.
- ALWAYS PREPEND '{os.environ["BASE_URL"]}' TO THE FILE PATHS IN THE RAW OUTPUT.
- For 'generated_code', include the code that was executed.
- For 'error', include any error message, or null if execution was successful.
- For 'selected_dataset', include the selected dataset name or null if none was used.
- For 'selected_datasets', include a list of datasets if multiple were used, or null if not applicable.
- For 'dataset_path_placeholder', include a suggested placeholder to help users reference this dataset in future queries.
- For 'output_directory_placeholder', include a suggested placeholder for the output directory.

Code output:
{{output}}

Generated code:
{{code}}

Error (if any):
{{error}}

Selected dataset:
{{dataset}}

Output a JSON object according to this schema:
{{format_instructions}}
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
    """
    logger.info("Top-level orchestrator started for user query:\n%s, file_input: %s", user_query, file_input)
    try:
        try:
            BASE_URL = os.environ["BASE_URL"]
        except KeyError:
            raise ValueError("BASE_URL environment variable is not set. Please set it in your .env file.")
            
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
                "BPD_Sun_normalized_log_deg.h5ad",
                "BPD_Sucre_normalized_log_deg.h5ad"
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

def explain_file_input():
    """Generate a document explaining how to use file input in code generation."""
    with open(os.path.join(os.path.dirname(__file__), "docs", "code_generation_file_input.md"), "w") as f:
        f.write(f"""# Using File Input with Code Generation

The Code Generator tool can analyze and process input files in several ways:

## File Input Types
* **File Upload**: Upload a local file directly through the chat interface for analysis.
* **File Path**: Provide a full path to a file on the server (must be within allowed directories).
* **URL**: Provide a URL to access a file (e.g., from previous plot generations).
* Can be a URL starting with {os.environ["BASE_URL"]}/ (will be automatically converted to a local path)

## Supported File Types
Currently supported file types include:
* TSV (.tsv) 
* CSV (.csv) 
* H5AD (.h5ad)

## Example Queries with File Input
* "Load this TSV file and count the number of rows"
* "Create a bar plot showing the distribution of values in column 'X' from this CSV"
* "Find the top 10 genes by expression in clusters from this h5ad file"

## Capabilities
The Code Generator can:
* Read and parse tabular data (TSV/CSV)
* Extract metadata from annotations
* Filter and transform data
* Generate statistical analyses
* Create custom visualizations

## Implementation Notes
* The tool automatically handles file path resolution and URL conversion
* Files are read using appropriate libraries (pandas for CSV/TSV, scanpy for h5ad)
* Error handling includes checking for file existence and valid file types
""")
    return "File input documentation generated."
