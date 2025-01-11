# LungMAP scExplore

## Overview
**LungMAP scExplore** is an advanced application for exploring and visualizing scRNA-seq datasets from LungMAP. It leverages conversational AI to provide interactive dataset exploration and visualization capabilities. Users can interact with the assistant via multiple interfaces, including a Gradio-based chatbot and a Chainlit-powered chat application. The system integrates various tools for dataset metadata retrieval, UMAP plot generation, and internet searches, all powered by OpenAI's GPT-based models.

---

## Main Interfaces and Files

### Gradio Chat Interface
- **Main Script:** `figchat_gradio.py`
  - Launches a Gradio-based chatbot interface for user interaction.
  - Loads environment variables, preloads datasets, and integrates key tools for exploration and visualization.
  - Allows users to query dataset information and generate visualizations like UMAP plots.

### Chainlit Chat Interface
- **Main Script:** `chainlit_app.py`
  - Sets up a Chainlit-based conversational interface.
  - Uses a state graph with LangGraph to manage conversation flow and tool invocation.
  - Provides similar functionalities as the Gradio interface with conversational state persistence.

### Documentation and Information
- **User Guide:** `chainlit.md`
  - Introduces LungMAP scExplore, detailing available datasets, features, and future capabilities.
  - Serves as a welcome page and documentation for researchers using the Chainlit interface.

### Tools and Utilities
- **`dataset_info_tool.py`**
  - Provides detailed metadata about available LungMAP datasets.
  - Parses dataset index files to return structured JSON with dataset descriptions, species, research teams, and more.

- **`visualization_tool.py`**
  - Handles visualization requests, primarily generating UMAP plots.
  - Loads datasets, computes UMAP embeddings, generates plots, and stores them in configured directories.
  - Integrates with OpenAI models to generate detailed descriptions of the plots.

- **`internet_search_tool.py`**
  - Performs internet searches using DuckDuckGo to retrieve additional information.
  - Useful for queries that extend beyond preloaded dataset capabilities.

- **`.env`**
  - Configuration file specifying environment variables such as API keys, dataset directories, output directories, dataset index file location, and database path.

- **`requirements.txt`**
  - Lists Python dependencies required to run LungMAP scExplore.

---

## Configuration (`.env`)
Ensure you have a `.env` file configured with the correct paths and API key:

```dotenv
OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
BASE_DATASET_DIR=/path/datasets
PLOT_OUTPUT_DIR=/path/plots
DATASET_INDEX_FILE=/path/dataset_index_advanced_paths.tsv
DATABASE_PATH=/path/global_sqlite.db
```

---

## Requirements
Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

---

## How to Run

### Gradio Interface
1. Clone the repository and navigate to the project directory:

    ```bash
    cd /path/to/LungMAP_scExplore
    ```
2. Pull a sqlite file:
    ```bash
    mkdir -p database && [ ! -f database/global_sqlite.db ] && wget -O database/global_sqlite.db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db
    ```
3. Set up the `.env` file with your paths and API key.
  
4. Ensure you are connected to the necessary network and have proxy settings configured if required.
   
5. Run the Gradio application:

    ```bash
    python figchat_gradio.py
    ```

7. Open the provided Gradio link in your browser to interact with LungMAP scExplore.

### Chainlit Interface
1. Follow steps 1–3 above for setup.

2. Run the Chainlit application:

    ```bash
    chainlit run chainlit_app.py
    ```

3. Follow the on-screen instructions to start chatting with LungMAP scExplore via the Chainlit interface.

---

## Functionality

- **Dataset Information:** Retrieve structured metadata about LungMAP datasets using `dataset_info_tool.py`.
- **Visualization:** Generate UMAP plots with options to color by cell type, disease, etc., using `visualization_tool.py`.
- **Internet Search:** Perform web searches for general queries with `internet_search_tool.py`.

---

## Example Queries

- "Tell me about the datasets available."
- "Generate a UMAP plot for the HLCA dataset colored by cell type."
- "Search the internet for recent publications on lung development."

---

## Future Capabilities

Planned enhancements include:

- Additional plot types such as heatmaps, violin plots, radar plots, and more.
- Comparative analysis tools for deeper insights.
- Enhanced conversation management and database integration for performance improvements.

---
