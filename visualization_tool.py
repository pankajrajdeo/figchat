# visualization_tool.py
import os
import json
import base64
from typing import Literal, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from utils import parse_tsv_data
from preload_datasets import PLOT_OUTPUT_DIR
import matplotlib
import logging
from matplotlib import rcParams
from figure_generation import main
import pandas as pd

BASE_URL = "https://devapp.lungmap.net"

# Suppress font-related messages
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

# Set a default font family to avoid Arial warnings
rcParams['font.family'] = 'DejaVu Sans'

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
       - Best for a concise overview of composition changes (e.g., “How do cell types differ by disease?”).

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
        - Use for requests involving “key transcriptional regulators,” “gene-gene interactions,” or regulatory relationships.

    11. **stats**
        - Outputs a TSV file summarizing differentially expressed genes (DEGs) for the specified condition, cell type, or disease.
        - Includes details like gene names, p-values, log fold changes, and associated metadata.
        - Use when the user requires a structured summary of DEGs for downstream analysis or validation.

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

Based on the DEG metadata, determine if DEGs exist for the specified disease and cell type combination in the user query. 
If DEGs exist, output:
- "deg_existence": true
Otherwise, output:
- "deg_existence": false
Include any suggestions or alternative options if DEGs do not exist.

Your output should be a JSON object adhering to this schema:
{format_instructions}
"""

deg_check_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", DEG_CHECK_PROMPT_TEMPLATE),
        ("human", "{user_query}")
    ]
).partial(format_instructions=degcheck_parser.get_format_instructions())

# Utility
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

def run_workflow1(user_query: str) -> Workflow1Model:
    dataset_metadata_str = get_dataset_metadata()
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    chain = workflow1_prompt | model | workflow1_parser
    result: Workflow1Model = chain.invoke({
        "user_query": user_query,
        "dataset_metadata": dataset_metadata_str
    })
    return result

def run_workflow2(user_query: str, selected_dataset: str, dataset_metadata_str: str) -> DEGCheckModel:
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

    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    chain = deg_check_prompt | model | degcheck_parser
    result: DEGCheckModel = chain.invoke({
        "user_query": user_query,
        "dataset_name": selected_dataset,
        "deg_metadata": deg_metadata
    })

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
    return result

specified_plots = {"volcano", "heatmap", "dotplot", "network", "stats"}  #plot types that can accept external gene symbols - heatmap, dotplot, network

###############################################################################
# Code 2: Workflow 3 (UNMODIFIED Prompt Template, Classes, Logic), plus local
# definition for get_single_dataset_metadata to avoid ImportError.
###############################################################################
import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

# Import specialized plot classes, PLOT_GUIDES, and utility functions from utils.py
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
    PLOT_GUIDES,
    parse_tsv_data  # parse_tsv_data is still needed
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

####################################
# Constants
####################################
DATASET_INDEX_FILE = "/data/aronow/pankaj/FigChat/datasets/dataset_index_advanced_paths.tsv"

####################################
# Workflow 3
####################################
THIRD_PROMPT_TEMPLATE = """\
You are a plot configuration assistant. The user wants to create a "{plot_type}". 
Your role is to generate a valid JSON configuration for the selected plot type by accurately interpreting the dataset metadata and correcting any errors in the user query.

Here is a description of how this plot type works and what arguments it needs:

Plot Type: {plot_type}
------------
{plot_description}

Below is the dataset metadata for the chosen dataset, which will help you determine valid field names, acceptable values, and constraints:
{dataset_metadata}

User Query (Refined):
{refined_query}

### Your Task:
1. **Use Dataset Metadata:** Explicitly match field names and values to the metadata provided. If a user-provided value does not match, look for the closest valid match in the metadata (e.g., correcting "capilary" to "CAP1" if "CAP1" is valid in the metadata).
   
2. **Correct Misspellings and Resolve Ambiguities:**
   - Correct gene names, disease names, and other terms using your internal knowledge. For example:
     - Correct "AGER1" to "AGER" if "AGER" is valid.
     - Resolve "interstitial lung diseases" to "interstitial lung disease" based on the metadata.
   - Disambiguate ambiguous terms using context and metadata.

3. **Generate Valid JSON:**
   - Return a JSON object that conforms exactly to the schema of the correct Pydantic class for "{plot_type}" in the codebase.
   - Exclude irrelevant or optional fields, unless specified in the user query or metadata.

4. **Strict Adherence to Schema:**
   - Use the correct field names, types, and defaults based on both the dataset metadata and the Pydantic model schema.
   - Only include fields relevant to "{plot_type}" from the code base.

5. **Output Format:**
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

def plot_config_generator(dataset_name: str, plot_type: str, refined_query: str) -> str:
    """
    Takes a dataset name, chosen plot type, and refined query to produce a single PlotConfig JSON,
    ready for figure_generation.py. This function wraps the Workflow 3 logic.
    """
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
            ("human", "{refined_query}")
        ]
    ).partial(
        format_instructions=parser.get_format_instructions()
    )

    prompt_text = prompt_template.format(
        plot_type=plot_type,
        plot_description=plot_description,
        dataset_metadata=dataset_metadata_str,
        refined_query=refined_query
    )

    # Call the LLM and parse output using the chosen Pydantic class
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    chain = prompt_template | model | parser

    result_config = chain.invoke({
        "refined_query": refined_query,
        "dataset_metadata": dataset_metadata_str,
        "plot_description": plot_description,
        "plot_type": plot_type
    })

    # Attempt to map the adata file path
    ADATA_FILE_PATH_MAP = {
        "HLCA_full_superadata_v3_norm_log_deg.h5ad": "/reference-data/figchat_datasets/HLCA_full_superadata_v3_norm_log_deg/HLCA_full_superadata_v3_norm_log_deg.h5ad",
        "HCA_fetal_lung_normalized_log_deg.h5ad": "/reference-data/figchat_datasets/HCA_fetal_lung_normalized_log_deg/HCA_fetal_lung_normalized_log_deg.h5ad",
        "BPD_infant_Sun_normalized_log_deg.h5ad": "/reference-data/figchat_datasets/BPD_infant_Sun_normalized_log_deg/BPD_infant_Sun_normalized_log_deg.h5ad",
        "BPD_fetal_normalized_log_deg.h5ad": "/reference-data/figchat_datasets/BPD_fetal_normalized_log_deg/BPD_fetal_normalized_log_deg.h5ad",
    }

    # Replace the adata_file path if recognized
    adata_file_name = os.path.basename(result_config.adata_file)
    if adata_file_name in ADATA_FILE_PATH_MAP:
        result_config.adata_file = ADATA_FILE_PATH_MAP[adata_file_name]
    else:
        raise ValueError(f"Unknown adata_file: {adata_file_name}. Please add it to the mapping.")

    # Convert the resulting pydantic model to valid JSON
    final_json = result_config.model_dump_json(indent=4)
    return final_json

###############################################################################
# Workflow 4: Image Description Generator
###############################################################################
def generate_image_description(image_path: str, plot_type: str) -> str:
    """
    Generates a description of the image using the existing prompt, dynamically including the plot type.

    Parameters:
    - image_path: Path to the image file.
    - plot_type: Type of the plot (e.g., 'heatmap', 'volcano', etc.).

    Returns:
    - Description of the image as a string.
    """
    from langchain_core.messages import HumanMessage
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    # Add plot_type to the prompt without changing the core template
    message = HumanMessage(
        content=[
            {"type": "text", "text": f"""
Provide a detailed description of the image, covering EACH AND EVERY textual and visual element comprehensively. The image represents a '{plot_type}' plot.
Summarize the overall structure and highlight key features such as cluster shapes, patterns, gradients, axes, legends, and titles. Describe trends, notable regions, or transitions, and explain how visual elements like colors relate to metadata (e.g., 'cell_type').

Use clear and concise language, appropriate for computational biologists, to explain terms like 'clusters' and 'dimensionality reduction.' Ensure the description flows logically, starting with an overview, diving into specific details, and concluding with interpretations of visual patterns and their potential biological relevance. NOTE: WRITE DETAIL ABOUT EACH AND EVERY CELL TYPE YOU SEE AND THEIR COLOURS AS WELL.
"""}, 
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    response = model.invoke([message])
    return response.content


###############################################################################
# Merged Functionality: Single "visualization_tool" that runs everything
###############################################################################
def visualization_tool(user_query: str) -> dict:
    """
    Identifies relevant datasets, identifies appropriate plot_types, parses plot arguments, generates the plots,
    and returns the output plot paths in JSON format.
    """

    # 1) Run Workflow 1
    print("=== Starting Workflow 1 ===")
    w1_result = run_workflow1(user_query)
    print("Workflow 1 completed. Results:", w1_result)

    # 2) Check if the plot type requires a DEG existence check
    if w1_result.plot_type in specified_plots:
        print("=== Plot type requires DEG check. Starting Workflow 2 ===")
        # Run Workflow 2
        dataset_metadata_str = get_dataset_metadata()
        w2_result = run_workflow2(user_query, w1_result.dataset_name, dataset_metadata_str)
        print("Workflow 2 completed. Results:", w2_result)

        # If deg_existence = false, stop and return
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
                config_json = plot_config_generator(w1_result.dataset_name, w1_result.plot_type, user_query)
                print("Workflow 3 completed. Config JSON generated:", config_json)

                # Write the config to a file
                os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
                output_json_path = os.path.join(PLOT_OUTPUT_DIR, "plot_config.json")
                with open(output_json_path, "w") as jf:
                    jf.write(config_json)

                # Execute figure_generation.py
                output_dir = PLOT_OUTPUT_DIR
                plot_outputs_raw = main(json_input=output_json_path, output_dir=output_dir)
                print("Figure generation outputs:", plot_outputs_raw)

                # Process multiple plot outputs with Workflow 4 integration
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

                final_output = {"plot_type": w1_result.plot_type.upper()}

                # Process each PNG through the description generator
                for i, (local_path, url) in enumerate(png_entries, start=1):
                    description = generate_image_description(local_path, w1_result.plot_type)
                    print(f"Generated description for {local_path}: {description}")
                    final_output[f"png_path_{i}"] = url
                    final_output[f"image_description_{i}"] = description

                # Add PDF paths to output
                for j, pdf in enumerate(pdf_paths, start=1):
                    final_output[f"pdf_path_{j}"] = pdf

                # Add TSV path if available
                if tsv_path:
                    final_output["tsv_path"] = tsv_path

                print("Final output ready:", final_output)
                return final_output

            except Exception as e:
                error_msg = f"Error in Workflow 3 or figure_generation: {repr(e)}"
                print(error_msg)
                return {"error": error_msg}

    # 3) If plot_type not in specified_plots, skip Workflow 2 and go directly to Workflow 3
    else:
        print("=== Plot type does not require DEG check. Starting Workflow 3 directly ===")
        try:
            config_json = plot_config_generator(w1_result.dataset_name, w1_result.plot_type, user_query)
            print("Workflow 3 completed. Config JSON generated:", config_json)

            # Write the config to a file
            os.makedirs(PLOT_OUTPUT_DIR, exist_ok=True)
            output_json_path = os.path.join(PLOT_OUTPUT_DIR, "plot_config.json")
            with open(output_json_path, "w") as jf:
                jf.write(config_json)

            # Execute figure_generation.py
            output_dir = PLOT_OUTPUT_DIR
            plot_outputs_raw = main(json_input=output_json_path, output_dir=output_dir)
            print("Figure generation outputs:", plot_outputs_raw)

            # Process multiple plot outputs with Workflow 4 integration
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

            final_output = {"plot_type": w1_result.plot_type.upper()}

            # Process each PNG through the description generator
            for i, (local_path, url) in enumerate(png_entries, start=1):
                description = generate_image_description(local_path)
                print(f"Generated description for {local_path}: {description}")
                final_output[f"png_path_{i}"] = url
                final_output[f"image_description_{i}"] = description

            # Add PDF paths to output
            for j, pdf in enumerate(pdf_paths, start=1):
                final_output[f"pdf_path_{j}"] = pdf

            # Add TSV path if available
            if tsv_path:
                final_output["tsv_path"] = tsv_path

            print("Final output ready:", final_output)
            return final_output

        except Exception as e:
            error_msg = f"Error in Workflow 3 or figure_generation: {repr(e)}"
            print(error_msg)
            return {"error": error_msg} 
