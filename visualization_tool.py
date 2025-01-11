# visualization_tool.py
import os
import uuid
import scanpy as sc
import matplotlib.pyplot as plt
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
import json
import base64

# Placeholders for globals to be set externally
PRELOADED_DATA = {}
PRELOADED_DATASET_INDEX = None
PLOT_OUTPUT_DIR = None

# Pydantic Models for Dataset Results
class DatasetResult(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset directory")
    h5ad_file: str = Field(..., description="Name of the .h5ad file")
    plot_type: str = Field(..., description="Type of plot requested")
    color_by: str = Field("cell_type", description="Observation column to color the plot")
    additional_args: Dict[str, Any] = Field({}, description="Additional arguments for the plot")

class DatasetResults(BaseModel):
    results: List[DatasetResult]

# Output parser
parser = PydanticOutputParser(pydantic_object=DatasetResults)

# Prompt Template for Dataset Routing
PROMPT_TEMPLATE = """\
You are a dataset router and plot configuration assistant. Based on the user's query, identify the most relevant dataset(s) and parse the required arguments for the requested plot type.

Dataset Metadata:
{dataset_metadata}

Each dataset includes:
- Name
- Description
- Directory Path
- h5ad File Name
- Cell Types
- Conditions/Diseases
- Assay Types
- Other metadata fields

For each matching dataset, return:
- `dataset_path`: Directory where the dataset resides
- `h5ad_file`: Name of the dataset file
- `plot_type`: The type of plot to generate (e.g., UMAP, heatmap, violin plot)
- `color_by`: The observation column to color the plot (default is "cell_type")
- `additional_args`: Any additional arguments required for the plot (default is an empty dictionary)

Respond with a JSON matching the schema:
{format_instructions}

User Query: {user_query}
"""

# Prepare the prompt
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", PROMPT_TEMPLATE),
        ("human", "{user_query}")
    ]
).partial(format_instructions=parser.get_format_instructions())

def load_dataset_index_as_string() -> str:
    """
    Return the preloaded dataset index as a string.
    """
    if PRELOADED_DATASET_INDEX is None:
        raise RuntimeError("Dataset index is not preloaded into memory.")
    
    return PRELOADED_DATASET_INDEX.to_string(index=False)

def generate_image_description(image_path: str) -> str:
    """
    Generates a description of the image using a multimodal input query.
    
    Parameters:
    - image_path: Path to the image file.

    Returns:
    - Description of the image as a string.
    """
    from langchain_core.messages import HumanMessage
    with open(image_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": """
Provide a detailed description of the image, covering EACH AND EVERY textual and visual elements comprehensively. Summarize the overall structure (e.g., a UMAP plot with clusters of cells) and highlight key features such as cluster shapes, patterns, gradients, axes, legends, and titles. Describe trends, notable regions, or transitions, and explain how visual elements like colors relate to metadata (e.g., 'cell_type').
Use clear and concise language, appropriate for computational biologists, to explain terms like 'clusters' and 'dimensionality reduction.' Ensure the description flows logically, starting with an overview, diving into specific details, and concluding with interpretations of visual patterns and their potential biological relevance. NOTE: WRITE DETAIL ABOUT EACH AND EVERY CELL TYPE YOU SEE AND THEIR COLOURS AS WELL.
"""}, 
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    )

    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    response = model.invoke([message])
    return response.content

import uuid  # Ensure this import is at the top of the file with other imports

def plot_umap(dataset_path: str, h5ad_file: str, color_by: str) -> Dict[str, str]:
    """
    Generates a UMAP plot for the given dataset and returns the paths of the plot files.

    Parameters:
    - dataset_path: Path to the dataset directory.
    - h5ad_file: Name of the .h5ad file.
    - color_by: The observation column to color the plot.

    Returns:
    - Dictionary containing paths to the PDF and PNG plot files.
    """
    fullpath = os.path.join(dataset_path, h5ad_file)

    # Check if the dataset is preloaded
    if fullpath in PRELOADED_DATA:
        adata = PRELOADED_DATA[fullpath]
    else:
        # Fallback to loading the dataset from disk
        if not os.path.exists(fullpath):
            raise FileNotFoundError(f"The file {fullpath} does not exist.")
        print(f"Loading {fullpath} from disk...")
        adata = sc.read_h5ad(fullpath)

    try:
        # Check if the specified column exists in the observations
        if color_by not in adata.obs.columns:
            raise ValueError(f"Column '{color_by}' not available in dataset {h5ad_file}.")

        # Ensure UMAP embeddings are computed
        if "X_umap" not in adata.obsm:
            sc.pp.neighbors(adata)
            sc.tl.umap(adata)

        # Ensure the output directory exists
        if not os.path.exists(PLOT_OUTPUT_DIR):
            os.makedirs(PLOT_OUTPUT_DIR)

        # Generate the UMAP plot
        plt.figure(figsize=(10, 8))
        sc.pl.umap(
            adata,
            color=color_by,
            show=False,
            title=f"UMAP colored by {color_by}"
        )

        # Generate a unique filename using UUID
        unique_id = uuid.uuid4().hex
        pdf_output_file = os.path.join(
            PLOT_OUTPUT_DIR, f"UMAP_{h5ad_file}_{color_by}_{unique_id}.pdf"
        )
        png_output_file = pdf_output_file.replace(".pdf", ".png")

        plt.savefig(pdf_output_file, bbox_inches="tight")
        plt.savefig(png_output_file, bbox_inches="tight", dpi=300)
        plt.close()

        # Return the paths to the plot files
        return {"pdf_path": pdf_output_file, "png_path": png_output_file}
    except Exception as e:
        raise RuntimeError(f"Error plotting UMAP: {e}")

def visualization_tool(user_query: str) -> str:
    """
    Identifies relevant datasets, parses plot arguments, generates the plots,
    and returns the output plot paths in JSON format.
    """
    try:
        dataset_metadata_str = load_dataset_index_as_string()
        model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
        chain = prompt | model | parser
        result: DatasetResults = chain.invoke({
            "user_query": user_query,
            "dataset_metadata": dataset_metadata_str
        })
        raw_output = json.dumps(result.model_dump(), indent=4)
        return parse_and_execute(raw_output)
    except Exception as e:
        return json.dumps({"error": f"Error during dataset parsing: {repr(e)}"}, indent=4)

def parse_and_execute(output: str) -> str:
    """
    Parses the JSON output, executes the appropriate plotting function,
    and returns a JSON object with the plot output paths and plot types.
    """
    try:
        BASE_URL = "http://devapp.lungmap.net"
        data = json.loads(output)
        plot_outputs = []

        for result in data.get("results", []):
            plot_type = result.get("plot_type", "").lower()
            dataset_path = result.get("dataset_path")
            h5ad_file = result.get("h5ad_file")
            color_by = result.get("color_by", "cell_type")
            
            if plot_type == "umap":
                try:
                    plot_paths = plot_umap(dataset_path, h5ad_file, color_by)
                    
                    # Generate image description with original local path
                    image_description = generate_image_description(plot_paths["png_path"])
                    
                    # Prepend BASE_URL to plot paths for JSON output
                    plot_outputs.append({
                        "plot_type": "UMAP",
                        "pdf_path": f"{BASE_URL}{plot_paths['pdf_path']}" if plot_paths.get("pdf_path") else None,
                        "png_path": f"{BASE_URL}{plot_paths['png_path']}" if plot_paths.get("png_path") else None,
                        "image_description": image_description
                    })
                except RuntimeError as e:
                    # Check if error message indicates missing column
                    if "Column '" in str(e) and "not available" in str(e):
                        return json.dumps({
                            "error": f"{str(e)} Please specify an alternative column (e.g., 'cell_type' or 'disease')."
                        }, indent=4)
                    else:
                        raise e
            else:
                print(f"Plot type '{plot_type}' is not supported.")

        # Return JSON with plot paths and types
        return json.dumps({"plots": plot_outputs}, indent=4)
    except Exception as e:
        return json.dumps({"error": f"Error during execution: {repr(e)}"}, indent=4)
