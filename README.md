# FigChat Application Deployment

## Overview
**FigChat** is an application for exploring and visualizing scRNA-seq datasets, specifically designed for LungMAP datasets. It provides a chatbot interface powered by Gradio and integrates tools for dataset exploration and visualization. The application uses OpenAI's GPT-based models for conversational interactions and advanced plot routing.

## Main File
The main script to execute is **`figchat_gradio.py`**, which initializes the application and provides a Gradio-based chatbot interface for user interaction.

### Features
- Dataset exploration through structured metadata queries using `dataset_info_tool.py`.
- Data visualization (e.g., UMAP plots) with customizable options using `visualization_tool.py`.
- Supports sandboxed links for downloading plots in various formats.
- Context-aware conversational interface for handling queries.
- Conversation history persistence in a SQLite database for quality control.

## File Descriptions
### `figchat_gradio.py`
The main file for launching the FigChat application. It:
- Loads environment variables from the `.env` file.
- Preloads datasets and metadata for quick access.
- Integrates tools for dataset exploration and visualization.
- Sets up the Gradio chatbot interface.

### `dataset_info_tool.py`
Provides detailed metadata about the available LungMAP datasets. It includes information such as:
- Dataset name
- Description
- Species
- Research team
- Publication and source
- Relevant covariates (e.g., age, sex, disease status)

### `visualization_tool.py`
Handles visualization requests, primarily generating UMAP plots. It includes:
- Plot generation based on user-specified parameters (e.g., color by cell type or disease).
- Error handling for missing observation columns.
- Plot storage in configurable output directories.
- Integration with OpenAI models for generating detailed descriptions of plots.

### `.env`
A configuration file that specifies global paths and API keys. Note that the provided paths reference directories on the **CCHMC network drive**. Ensure that you have access to the network drive, and if you're using the application from within the CCHMC network, **remember to enable the proxy** to allow access to the OpenAI API.

#### Configuration Variables:
- `OPENAI_API_KEY`: Your OpenAI API key for generating conversational responses.
- `BASE_DATASET_DIR`: Path to the directory containing datasets.
- `PLOT_OUTPUT_DIR`: Path for storing generated plots.
- `DATASET_INDEX_FILE`: Path to the dataset index file.
- `DATABASE_PATH`: Path to the SQLite database for conversation persistence.

Example `.env` file:
```
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
BASE_DATASET_DIR=/data/aronow/pankaj/FigChat/datasets
PLOT_OUTPUT_DIR=/data/aronow/pankaj/FigChat/run_code/plots
DATASET_INDEX_FILE=/data/aronow/pankaj/FigChat/datasets/dataset_index_advanced_paths.tsv
DATABASE_PATH=/data/aronow/pankaj/FigChat/database/global_sqlite.db
```

## Requirements
Install the required Python libraries using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## How to Run
1. Clone the repository and navigate to the project directory:
   ```bash
   cd /path/to/figchat_app_deployment
   ```

2. Set up the `.env` file with your paths and API key.

3. Ensure that you are connected to the CCHMC network drive. If using the application from within the CCHMC network, enable the proxy to ensure OpenAI API access.

4. Run the application:
   ```bash
   python figchat_gradio.py
   ```

5. Open the provided Gradio link in your browser to interact with FigChat.

## Functionality
- **Dataset Information**: Queries about dataset metadata are handled using the `dataset_info_tool`.
- **Visualization**: Generate UMAP plots by specifying the dataset and coloring options (e.g., cell type, disease). This is handled by `visualization_tool`

Example Queries:
- *"Tell me about the datasets available."*
- *"Generate a UMAP plot for HLCA dataset colored by cell type."*

## Future Capabilities
The application is designed for scalability. Planned features include:
- Additional plot types (e.g., heatmaps, violin plots, radar plots).
- Enhanced conversation management for cost optimization.
- Integration with PostgreSQL for improved database performance.
