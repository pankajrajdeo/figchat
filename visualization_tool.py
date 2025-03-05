# visualization_tool.py
import os
import json
import base64
import threading  # Added for thread-safe logging
from typing import Literal, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from utils import parse_tsv_data
from preload_datasets import (
    PLOT_OUTPUT_DIR,
    DATASET_INDEX_FILE,
    db
)
import matplotlib
matplotlib.use("Agg")
import logging
from matplotlib import rcParams
from figure_generation import main
import pandas as pd
from langchain.schema import SystemMessage, HumanMessage

BASE_URL = "https://devapp.lungmap.net"

# Define training data file path in standardized location
BASE_DATASET_DIR = os.environ.get("BASE_DATASET_DIR", "")
TRAIN_DATA_FILE = os.path.join(BASE_DATASET_DIR, "training_data", "visualization_tool_training_data.json")

# -----------------------------
# LLM
# -----------------------------
visualization_tool_llm_base = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
visualization_tool_llm_advanced = ChatOpenAI(model="gpt-4o")

# Suppress font-related messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Set a default font family to avoid Arial warnings
rcParams['font.family'] = 'DejaVu Sans'

# -----------------------------
# Logging Utility
# -----------------------------

# Initialize a lock for thread-safe file operations
log_lock = threading.Lock()

def load_log() -> dict:
    """
    Load the existing log from the TRAIN_DATA_FILE.
    If the file doesn't exist, initialize with an empty structure.
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(TRAIN_DATA_FILE), exist_ok=True)
    
    if not os.path.exists(TRAIN_DATA_FILE):
        return {"workflows": []}

    try:
        with open(TRAIN_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted, reset it
        return {"workflows": []}

def append_log(workflow_name: str, prompt: str, response: str) -> None:
    """
    Append a new log entry to the TRAIN_DATA_FILE.
    
    Parameters:
    - workflow_name: Name of the workflow (e.g., 'Workflow1').
    - prompt: The prompt sent to the LLM.
    - response: The response received from the LLM.
    """
    with log_lock:  # Ensure thread-safe access
        log_data = load_log()
        log_entry = {
            "workflow": workflow_name,
            "prompt": prompt,
            "response": response,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        log_data["workflows"].append(log_entry)
        with open(TRAIN_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

# -----------------------------
# Workflow 1: Dataset & Plot Selection
# -----------------------------

class Workflow1Model(BaseModel):
    dataset_name: Optional[Literal[
        "HLCA_full_superadata_v3_norm_log_deg.h5ad",
        "HCA_fetal_lung_normalized_log_deg.h5ad",
        "BPD_infant_Sun_normalized_log_deg.h5ad",
        "BPD_fetal_normalized_log_deg.h5ad"
    ]]
    reason: str
    plot_type: Literal[
        "heatmap", "radar", "cell_frequency", "volcano", "stats",
        "dotplot", "violin", "venn", "upset_genes", "umap", "network"
    ]
    is_marker_genes: bool

workflow1_parser = PydanticOutputParser(pydantic_object=Workflow1Model)

WORKFLOW1_PROMPT_TEMPLATE = """\
Based on the user's query and the available dataset metadata, perform the following tasks:

1. **Dataset Selection:**
   - Select the most relevant dataset from the available options.

2. **Plot Type Selection:**
   - Determine the most suitable plot type for visualizing the data in the context of the user's query.
   - Use the following plot type guidelines to assist your selection:

    1. **heatmap**
       - Shows expression levels (e.g., top genes, marker genes) across cells or aggregated cell groups.
       - Use for visualizing patterns/clusters or comparing expression intensities across multiple genes and/or cell types.

    2. **radar**
       - Displays average cell-type frequencies (proportions) across different conditions in a radial/spider chart.
       - Best for a concise overview of composition changes (e.g., "How do cell types differ by disease?").

    3. **cell_frequency**
       - Shows per-donor box/violin plots of cell-type frequencies with statistical tests (e.g., Mann-Whitney U).
       - Useful for detailed donor-level comparisons or p-values for differences in cell-type proportions by condition.

    4. **volcano**
       - A scatter plot of Log2(Fold Change) vs. -log10(p-value) to highlight significantly changed genes.
       - Use to visualize genes up/down-regulated in a comparison.

    5. **dotplot**
       - Compares multiple genes across groups/cell types.
       - Dot size reflects the fraction of cells expressing the gene; color reflects average expression.
       - Ideal for a quick overview of how specific genes are expressed across cell types/conditions.

    6. **violin**
       - Displays the distribution of expression for a single gene across conditions or groups.
       - Useful for focusing on one gene and its distribution.

    7. **venn**
       - Compares overlapping genes (e.g., shared DEGs) among up to 2-3 sets.
       - Relevant for explicitly requested overlapping gene sets.

    8. **upset_genes**
       - Compares overlaps among >2 gene sets or more complex intersections.
       - Similar to Venn but handles multiple groups more clearly.

    9. **umap**
       - Plots all cells in a 2D UMAP embedding, colored by cell type/cluster or expression of a single gene.
       - Use for an overview of cell clusters or spatial distribution of one gene across the manifold.

    10. **network**
        - Displays gene interaction or regulatory networks.
        - Use for requests involving "key transcriptional regulators," "gene-gene interactions," or regulatory relationships.

    11. **stats**
        - Outputs a TSV file summarizing differentially expressed genes (DEGs) for the specified condition, cell type, or disease.
        - Includes details like gene names, p-values, log fold changes, and associated metadata.
        - Use when the user requires a structured summary of DEGs for downstream analysis or validation.

3. **Marker Genes Detection:**
   - Determine if the user's query specifically mentions "marker genes."

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

PRELOADED_DATASET_INDEX = None
from preload_datasets import DATASET_INDEX_FILE

def get_dataset_metadata() -> str:
    global PRELOADED_DATASET_INDEX, DATASET_INDEX_FILE
    if PRELOADED_DATASET_INDEX is None:
        PRELOADED_DATASET_INDEX = parse_tsv_data(DATASET_INDEX_FILE)
    # Check if it's a DataFrame and convert if necessary
    if isinstance(PRELOADED_DATASET_INDEX, pd.DataFrame):
        data_to_dump = PRELOADED_DATASET_INDEX.to_dict(orient='list')
    else:
        data_to_dump = PRELOADED_DATASET_INDEX
    return json.dumps(data_to_dump, indent=4)

async def run_workflow1(user_query: str) -> Workflow1Model:
    dataset_metadata_str = get_dataset_metadata()
    model = visualization_tool_llm_base
    chain = workflow1_prompt | model | workflow1_parser
    
    # Format the prompt
    prompt_input = {
        "user_query": user_query,
        "dataset_metadata": dataset_metadata_str
    }
    formatted_prompt = workflow1_prompt.format(**prompt_input)
    
    # Invoke the chain asynchronously and get the response
    result = await chain.ainvoke(prompt_input)
    
    # Log the interaction
    append_log(
        workflow_name="Workflow1",
        prompt=formatted_prompt,
        response=result.model_dump_json()
    )
    
    return result

# -----------------------------
# Workflow 2: DEG Existence Confirmation
# -----------------------------

class DEGCheckModel(BaseModel):
    deg_existence: bool
    suggestion: Optional[str]

degcheck_parser = PydanticOutputParser(pydantic_object=DEGCheckModel)

DEG_CHECK_PROMPT_TEMPLATE = """\
You are an assistant specialized in confirming the existence of differentially expressed genes (DEGs) given a dataset's DEG metadata.

Dataset Name: {dataset_name}
Differential Expression Metadata:
{deg_metadata}

User Query:
{user_query}

Based on the DEG metadata, look very carefully at each field and determine if DEGs exist for the specified disease and cell type combination in the user query. 
If DEGs exist, output:
- "deg_existence": true
Otherwise, output:
- "deg_existence": false
Include any suggestions or alternative options if DEGs do not exist.

**Note**: Whenever cell type markers are mentioned, only confirm that the cell type exists and ignore any disease fields for DEGs.

Your output should be a JSON object adhering to this schema:
{format_instructions}
"""

deg_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DEG_CHECK_PROMPT_TEMPLATE),
        ("human", "{user_query}")
    ]
).partial(format_instructions=degcheck_parser.get_format_instructions())

async def run_workflow2(user_query: str, selected_dataset: str, dataset_metadata_str: str) -> DEGCheckModel:
    # Parse metadata to extract DEG information for the selected dataset
    all_metadata = json.loads(dataset_metadata_str)
    deg_metadata = "[]"
    alt_degs = []
    for ds in all_metadata.get("datasets", []):
        ds_name = ds.get("**Dataset Metadata**", {}).get("**Dataset Name**")
        if ds_name == selected_dataset:
            deg_info = ds.get("**Differential Expression** (Disease-Study-CellType mappings)", [])
            deg_metadata = json.dumps(deg_info, indent=2)
            alt_degs = deg_info
            break

    model = visualization_tool_llm_base
    chain = deg_check_prompt | model | degcheck_parser
    
    # Prepare the prompt input
    prompt_input = {
        "user_query": user_query,
        "dataset_name": selected_dataset,
        "deg_metadata": deg_metadata
    }
    formatted_prompt = deg_check_prompt.format(**prompt_input)
    
    # Invoke the chain asynchronously and get the response
    result = await chain.ainvoke(prompt_input)
    
    # Post-process suggestion
    if result.deg_existence:
        result.suggestion = None
    else:
        if alt_degs:
            result.suggestion = (
                "The DEGs for your specified disease/cell type are not available. "
                "Please choose from the following options: " + json.dumps(alt_degs, indent=4)
            )
        else:
            result.suggestion = "No DEG information available for the selected dataset."
    
    # Log the interaction
    append_log(
        workflow_name="Workflow2",
        prompt=formatted_prompt,
        response=result.model_dump_json()
    )
    
    return result

specified_plots = {"volcano", "heatmap", "dotplot", "network", "stats"}  # Plot types that require DEG check

# -----------------------------
# Workflow 3: Plot Configuration
# -----------------------------
from utils import (
    HeatmapPlotConfig,
    StatsPlotConfig,
    RadarPlotConfig,
    CellFrequencyPlotConfig,
    VolcanoPlotConfig,
    DotPlotConfig,
    ViolinPlotConfig,
    VennPlotConfig,
    UpSetGenesPlotConfig,
    UmapPlotConfig,
    NetworkPlotConfig,
    PLOT_GUIDES
)

def get_single_dataset_metadata(all_metadata: dict, target_dataset: str) -> dict:
    """
    Filters metadata to include only the relevant dataset entry.
    """
    filtered = {
        "datasets": [
            d for d in all_metadata.get("datasets", [])
            if d["**Dataset Metadata**"]["**Dataset Name**"] == target_dataset
        ],
        "notes": all_metadata.get("notes", {})
    }
    return filtered

THIRD_PROMPT_TEMPLATE = """\
You are a plot configuration assistant. The user wants to create a "{plot_type}". 
Your role is to generate a valid JSON configuration for the selected plot type by accurately interpreting the dataset metadata and correcting any errors in the user query.

Dataset Name: {dataset_name}

Here is a description of how this plot type works and what arguments it needs:

Plot Type: {plot_type}
------------
{plot_description}

Below is the dataset metadata for the chosen dataset, which will help you determine valid field names, acceptable values, and constraints:
{dataset_metadata}

User Query:
{user_query}

### Your Task:
1. **Use Dataset Metadata:** Explicitly match field names and values to the metadata provided. If a user-provided value does not match, look for the closest valid match in the metadata (e.g., correcting "capilary" to "CAP1" if "CAP1" is valid in the metadata).

2. **Correct Misspellings and Resolve Ambiguities:**
   - Correct gene names, disease names, and other terms using your internal knowledge. For example:
     - Correct "AGER1" to "AGER" if "AGER" is valid.
     - Resolve "interstitial lung diseases" to "interstitial lung disease" based on the metadata.
   - Disambiguate ambiguous terms using context and metadata.

3. **Derive Restrict Variables:**
   - Analyze the dataset metadata to identify fields that can be used as **restrict variables**.
   - Restrict variables are used to filter data for plotting and may include fields such as `study`, `sex`, `age`, `tissue`, `disease`, or other categorical variables.
   - Ensure the index field is present in the dataset metadata specified in the query. (e.g., if the user specifies a study by first author, you need to determine the correct study name from the metadata.)
   - Include valid restrict variables as part of the JSON configuration.

4. **Generate Valid JSON:**
   - Return a JSON object that conforms exactly to the schema of the correct Pydantic class for "{plot_type}" in the codebase.
   - Exclude irrelevant or optional fields, unless specified in the user query or metadata.
   - Make sure the file name you provide ends with .h5ad

5. **Strict Adherence to Schema:**
   - Use the correct field names, types, and defaults based on both the dataset metadata and the Pydantic model schema.
   - Only include fields relevant to "{plot_type}" from the code base.

6. **Output Format:**
   - Provide your output as a JSON object, unwrapped by Markdown formatting.
   - If a correction or adjustment is made, ensure the final JSON reflects the valid and corrected configuration.

#### JSON Configuration Schema (for reference):
{format_instructions}
"""

def get_plot_class(plot_type: str):
    """Map the plot_type string to the correct specialized Pydantic class."""
    if plot_type == "heatmap":
        return HeatmapPlotConfig
    elif plot_type == "stats":
        return StatsPlotConfig
    elif plot_type == "radar":
        return RadarPlotConfig
    elif plot_type == "cell_frequency":
        return CellFrequencyPlotConfig
    elif plot_type == "volcano":
        return VolcanoPlotConfig
    elif plot_type == "dotplot":
        return DotPlotConfig
    elif plot_type == "violin":
        return ViolinPlotConfig
    elif plot_type == "venn":
        return VennPlotConfig
    elif plot_type == "upset_genes":
        return UpSetGenesPlotConfig
    elif plot_type == "umap":
        return UmapPlotConfig
    elif plot_type == "network":
        return NetworkPlotConfig
    else:
        raise ValueError(f"Unsupported or unknown plot_type: {plot_type}")

async def plot_config_generator(dataset_name: str, plot_type: str, user_query: str) -> str:
    """
    Generates the configuration for a plot based on the dataset, plot type, and user query.
    Includes conditional logic for handling restrict_studies and study_index for specific plot types.
    """
    # Validate dataset_name
    if not dataset_name:
        raise ValueError("Dataset name is required but not provided to Workflow 3.")

    # 1) Load only the relevant metadata for the chosen dataset
    all_metadata = parse_tsv_data(DATASET_INDEX_FILE)
    single_dataset_metadata = get_single_dataset_metadata(all_metadata, dataset_name)
    dataset_metadata_str = json.dumps(single_dataset_metadata, indent=4)

    # 2) Retrieve the correct Pydantic class and plot description
    chosen_class = get_plot_class(plot_type)
    plot_description = PLOT_GUIDES.get(plot_type, "No guidance available for this plot type.")

    # 3) Build a parser for the chosen class
    parser = PydanticOutputParser(pydantic_object=chosen_class)

    # 4) Create the final prompt using the THIRD_PROMPT_TEMPLATE
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", THIRD_PROMPT_TEMPLATE),
            ("human", "{user_query}")
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    # Add the dataset_name to the prompt formatting
    prompt_text = prompt_template.format(
        plot_type=plot_type,
        dataset_name=dataset_name,
        plot_description=plot_description,
        dataset_metadata=dataset_metadata_str,
        user_query=user_query
    )

    # Use visualization_tool_llm_advanced only for heatmap and dotplot; otherwise use visualization_tool_llm_base.
    if plot_type in {"heatmap", "dotplot"}:
        model = visualization_tool_llm_advanced
    else:
        model = visualization_tool_llm_base

    # Call the LLM and parse output using the chosen Pydantic class
    chain = prompt_template | model | parser

    # Invoke the chain asynchronously
    result_config = await chain.ainvoke({
        "user_query": user_query,  # Changed from refined_query to user_query
        "dataset_metadata": dataset_metadata_str,
        "dataset_name": dataset_name,
        "plot_description": plot_description,
        "plot_type": plot_type
    })

    # Map the `adata_file` path and apply hardcoded logic for specific plot types
    ADATA_FILE_PATH_MAP = {
        "HLCA_full_superadata_v3_norm_log_deg.h5ad": os.path.join(os.environ.get("BASE_DATASET_DIR"), "HLCA_full_superadata_v3_norm_log_deg", "HLCA_full_superadata_v3_norm_log_deg.h5ad"),
        "HCA_fetal_lung_normalized_log_deg.h5ad": os.path.join(os.environ.get("BASE_DATASET_DIR"), "HCA_fetal_lung_normalized_log_deg", "HCA_fetal_lung_normalized_log_deg.h5ad"),
        "BPD_infant_Sun_normalized_log_deg.h5ad": os.path.join(os.environ.get("BASE_DATASET_DIR"), "BPD_infant_Sun_normalized_log_deg", "BPD_infant_Sun_normalized_log_deg.h5ad"),
        "BPD_fetal_normalized_log_deg.h5ad": os.path.join(os.environ.get("BASE_DATASET_DIR"), "BPD_fetal_normalized_log_deg", "BPD_fetal_normalized_log_deg.h5ad"),
    }

    # Replace the adata_file path if recognized
    adata_file_name = os.path.basename(result_config.adata_file)
    if adata_file_name in ADATA_FILE_PATH_MAP:
        result_config.adata_file = ADATA_FILE_PATH_MAP[adata_file_name]
    else:
        raise ValueError(f"Unknown adata_file: {adata_file_name}. Please add it to the mapping.")

    # Hardcoded logic for specific plot types and dataset
    if plot_type in {"dotplot", "cell_frequency", "radar", "heatmap"}:
        if adata_file_name == "HLCA_full_superadata_v3_norm_log_deg.h5ad":
            # For HLCA_full_superadata_v3_norm_log_deg.h5ad, use default values unless overridden by user
            if result_config.restrict_studies is None and result_config.covariates == ["normal"]:
                result_config.restrict_studies = ["Sun_2020"]
                result_config.study_index = "study"
            elif result_config.restrict_studies == ["Sun_2020"]:
                if result_config.covariates != ["normal"]:
                    result_config.restrict_studies = None
        else:
            # For other datasets, ensure these fields are null
            result_config.restrict_studies = None
            result_config.study_index = None

    # Convert the resulting Pydantic model to valid JSON
    final_json = result_config.model_dump_json(indent=4)
    
    # Log the interaction
    append_log(
        workflow_name="Workflow3",
        prompt=prompt_text,
        response=final_json
    )
    
    return final_json

# -----------------------------
# Merged Functionality: Single "visualization_tool" that runs everything
# -----------------------------
async def Data_Visualizer(user_query: str) -> dict:
    """
    Identifies relevant datasets, identifies appropriate plot_types, parses plot arguments, generates the plots,
    and returns the output plot paths in JSON format, including restrict_studies in the output.
    """
    try:
        # 1) Run Workflow 1
        print("=== Starting Workflow 1 ===")
        w1_result = await run_workflow1(user_query)
        print("Workflow 1 completed. Results:", w1_result)
        
        # 2) Check if the plot type requires a DEG existence check
        # Also, check if the query is about marker genes
        if w1_result.plot_type in specified_plots and not w1_result.is_marker_genes:
            print("=== Plot type requires DEG check. Starting Workflow 2 ===")
            dataset_metadata_str = get_dataset_metadata()
            w2_result = await run_workflow2(user_query, w1_result.dataset_name, dataset_metadata_str)
            print("Workflow 2 completed. Results:", w2_result)
            if not w2_result.deg_existence:
                print("DEG existence check failed. Terminating pipeline.")
                return {
                    "output": "deg=false",
                    "dataset_name": w1_result.dataset_name,
                    "plot_type": w1_result.plot_type,
                    "reason": w1_result.reason,
                    "deg_existence": w2_result.deg_existence,
                    "suggestion": w2_result.suggestion
                }
            else:
                print("DEG existence confirmed. Proceeding to Workflow 3.")
                try:
                    # Use the original user_query as the refined_query
                    config_json = await plot_config_generator(w1_result.dataset_name, w1_result.plot_type, user_query)
                    print("Workflow 3 completed. Config JSON generated:", config_json)
                    config_data = json.loads(config_json)
                    restrict_studies = config_data.get("restrict_studies")
                    os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
                    output_json_path = os.path.join(PLOT_OUTPUT_DIR, "plot_config.json")
                    with open(output_json_path, "w") as jf:
                        jf.write(config_json)
                    output_dir = PLOT_OUTPUT_DIR
                    plot_outputs_raw = main(json_input=output_json_path, output_dir=output_dir)
                    print("Figure generation outputs:", plot_outputs_raw)
                    png_entries = []
                    pdf_paths = []
                    tsv_path = None
                    for plot in plot_outputs_raw:
                        if isinstance(plot, list) and len(plot) == 2:
                            file_path = plot[0]
                            if file_path.endswith(".pdf"):
                                pdf_paths.append(f"{BASE_URL}{file_path}")
                            elif file_path.endswith(".png"):
                                png_entries.append((file_path, f"{BASE_URL}{file_path}"))
                            elif file_path.endswith(".tsv"):
                                tsv_path = f"{BASE_URL}{file_path}"
                        else:
                            raise ValueError(f"Unexpected plot format: {plot}")
                    final_output = {
                        "plot_type": w1_result.plot_type.upper(),
                        "restrict_studies (restricted to following study/studies)": restrict_studies,
                    }
                    for i, (local_path, url) in enumerate(png_entries, start=1):
                        final_output[f"png_path_{i}"] = url
                    for j, pdf in enumerate(pdf_paths, start=1):
                        final_output[f"pdf_path_{j}"] = pdf
                    if tsv_path:
                        final_output["tsv_path"] = tsv_path
                    final_output["generated_config"] = config_data
                    print("Final output ready:", final_output)
                    return final_output
                except Exception as e:
                    error_msg = f"Error in Workflow 3 or figure_generation: {repr(e)}"
                    print(error_msg)
                    return {"error": error_msg}
        else:
            if w1_result.is_marker_genes:
                print("=== Query specifies marker genes. Bypassing Workflow 2 and starting Workflow 3 ===")
            else:
                print("=== Plot type does not require DEG check. Starting Workflow 3 directly ===")
            try:
                # Use the original user_query as the refined_query
                config_json = await plot_config_generator(w1_result.dataset_name, w1_result.plot_type, user_query)
                print("Workflow 3 completed. Config JSON generated:", config_json)
                config_data = json.loads(config_json)
                restrict_studies = config_data.get("restrict_studies")
                os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
                output_json_path = os.path.join(PLOT_OUTPUT_DIR, "plot_config.json")
                with open(output_json_path, "w") as jf:
                    jf.write(config_json)
                output_dir = PLOT_OUTPUT_DIR
                plot_outputs_raw = main(json_input=output_json_path, output_dir=output_dir)
                print("Figure generation outputs:", plot_outputs_raw)
                png_entries = []
                pdf_paths = []
                tsv_path = None
                for plot in plot_outputs_raw:
                    if isinstance(plot, list) and len(plot) == 2:
                        file_path = plot[0]
                        if file_path.endswith(".pdf"):
                            pdf_paths.append(f"{BASE_URL}{file_path}")
                        elif file_path.endswith(".png"):
                            png_entries.append((file_path, f"{BASE_URL}{file_path}"))
                        elif file_path.endswith(".tsv"):
                            tsv_path = f"{BASE_URL}{file_path}"
                    else:
                        raise ValueError(f"Unexpected plot format: {plot}")
                final_output = {
                    "plot_type": w1_result.plot_type.upper(),
                    "restrict_studies (restricted to following study/studies)": restrict_studies,
                }
                for i, (local_path, url) in enumerate(png_entries, start=1):
                    final_output[f"png_path_{i}"] = url
                for j, pdf in enumerate(pdf_paths, start=1):
                    final_output[f"pdf_path_{j}"] = pdf
                if tsv_path:
                    final_output["tsv_path"] = tsv_path
                final_output["generated_config"] = config_data
                print("Final output ready:", final_output)
                return final_output
            except Exception as e:
                error_msg = f"Error in Workflow 3 or figure_generation: {repr(e)}"
                print(error_msg)
                return {"error": error_msg}
    except Exception as e:
        error_msg = f"Error in visualization pipeline: {repr(e)}"
        print(error_msg)
        return {"error": error_msg}

# For backward compatibility
visualization_tool = Data_Visualizer
