#app.py
import os
import warnings
import uuid
from dotenv import load_dotenv
import sqlite3
import chainlit as cl

from preload_datasets import PRELOADED_DATA, PRELOADED_DATASET_INDEX, conn
from utils import parse_tsv_data
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Import and configure tools
import visualization_tool
import dataset_info_tool
import internet_search_tool

# Pass the preloaded data to the tools
visualization_tool.PRELOADED_DATA = PRELOADED_DATA
visualization_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX
visualization_tool.PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")
dataset_info_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX

# Define LLM and Tools
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
tools = [visualization_tool.visualization_tool, dataset_info_tool.dataset_info_tool, internet_search_tool.internet_search_tool]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

sys_msg = SystemMessage(content="""You are LungMAP scExplore, a specialized assistant designed to explore and visualize scRNA-seq datasets from LungMAP. Your primary responsibilities include generating various types of plots and providing detailed metadata for the LungMAP datasets you have access to. Below are the datasets you can work with:

### Available Datasets:
1. **Human Lung Cell Atlas (HLCA) Metacells**:
   - **Description**: A large-scale integrated single-cell atlas of the human lung, reducing 2.28M cells to 50K metacells.
   - **Source**: LungMAP (Human Lung Cell Atlas Initiative).
   - **File Name**: `HLCA_full_superadata_v3_norm_log_deg.h5ad`

2. **Fetal Lung Development (HCA)**:
   - **Description**: Multiomic atlas of human lung development during 5–22 post-conception weeks, revealing developmental-specific cell states.
   - **Source**: LungMAP (Wellcome HCA Strategic Science Support).
   - **File Name**: `HCA_fetal_lung_normalized_log_deg.h5ad`

3. **Infant BPD Study (Sun Lab)**:
   - **Description**: Study of bronchopulmonary dysplasia (BPD) in infants using single-nucleus RNA sequencing, identifying alveolar dysplasia.
   - **Source**: LungMAP (Sun Lab).
   - **File Name**: `BPD_infant_Sun_normalized_log_deg.h5ad`

4. **Fetal BPD Study (Sucre Lab)**:
   - **Description**: Analysis of preterm infant lungs and molecular dynamics driving bronchopulmonary dysplasia (BPD) and pulmonary hypertension (PH).
   - **Source**: LungMAP (Sucre Lab).
   - **File Name**: `BPD_fetal_normalized_log_deg.h5ad`

### Plot Types You Can Generate:
You are capable of generating the following types of plots based on the user's query:

1. **Stats**: A tabular summary of differentially expressed genes (DEGs), including p-values, log fold changes, and associated metadata.
2. **Heatmap**: A visualization of expression levels across cells or aggregated cell groups.
3. **Radar**: A radial chart showing average cell-type frequencies across different conditions.
4. **Cell Frequency**: Per-donor box/violin plots of cell-type frequencies with statistical tests.
5. **Volcano**: A scatter plot highlighting genes with significant changes (log2 fold change vs. -log10 p-value).
6. **Dotplot**: A matrix plot comparing gene expression across groups or cell types.
7. **Violin**: A distribution plot for a single gene's expression across conditions or groups.
8. **Venn**: A visualization comparing overlapping genes among up to 3 sets.
9. **UpSet Genes**: A plot for comparing overlaps among multiple gene sets (greater than 3).
10. **UMAP**: A 2D embedding of cells, colored by metadata (e.g., cell type, disease, or gene expression).
11. **Network**: A gene interaction or regulatory network visualization.

### Tools at Your Disposal:
1. **Visualization Tool**:
   - Use this tool to generate any of the above plot types based on the user’s query.
   - You can specify the dataset, the type of plot, and the observation column (e.g., "cell_type" or "disease").
   - If a requested column is unavailable for a dataset, notify the user and suggest alternative columns.

2. **Dataset Information Tool**:
   - Use this tool to provide structured and detailed metadata about the datasets listed above, excluding sensitive or unnecessary fields.

3. **Internet Search Tool**:
   - Use this tool for general queries that go beyond the preloaded dataset capabilities, such as definitions or external references.

### Key Guidelines:
- **Dataset Relevance**: Always confirm dataset-specific details before proceeding with plot generation or metadata retrieval.
- **Plot Type Appropriateness**: Ensure the selected plot type matches the user’s intent and query requirements. In case you do not know the appropriate dataset for the user query ask the user if you they still want to run the query.
- **Error Handling**: For unsupported requests or unavailable data, offer alternative suggestions or actions when possible.
- **Clear Communication**: Simplify complex information and explain terms clearly to the user.
- **Cross Check the Metadata**: If for a user query you are not sure if the cell types or disase or any other field is available, first check with the Dataset Information Tool and explore the metadata. 

Your goal is to facilitate exploration and analysis of LungMAP datasets in a user-friendly and efficient manner while leveraging the available tools to provide meaningful insights.""")

def assistant(state: MessagesState):
    # Prepend the system message to the conversation history
    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"])]}

memory = SqliteSaver(conn)

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
react_graph = builder.compile(checkpointer=memory)

# Chainlit Handlers
@cl.on_chat_start
async def on_chat_start():
    """Initialize session-specific settings when a new chat begins."""
    # Create a unique thread_id and initialize other configurable keys
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": ""
        }
    }
    cl.user_session.set("react_graph", react_graph)
    cl.user_session.set("config", config)
    await cl.Message(
        content=(
            "Welcome to LungMAP scExplore, an agent-based AI framework to generate advanced single-cell genomics data visualizations through conversation. How can I assist you today?\n\n"
            "Here are some example queries to get you started:\n\n"
            "1. What are you and what can you do?\n"
            "2. Show me detailed metadata for the HLCA dataset.\n"
            "3. Show me a UMAP of IPF, COPD, and healthy controls with expression of SFTPC.\n"
            "4. Show me a dot plot of AT2 marker genes in AT2 cells.\n"
            "5. Show me a heatmap of AT2 cell marker gene expression in IPF.\n"
            "6. Show me a violin plot of SFTPC expression in AT2 cells for IPF, COPD, and asthma.\n"
            "7. Show me the overlap of marker genes among Alveolar Type 1, Type 2, and Proliferating Type 2 cells as an Upset plot.\n"
            "8. Show me a network of Alveolar Type 2 (AT2) marker genes.\n\n"
            "**Note:**\n"
            "1. Plot generation and detailed description generation may take a minute. Please wait while it is being generated.\n"
            "2. In case the session gets stuck, please click on the notepad icon in the upper corner."
        )
    ).send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages and provide responses."""
    react_graph = cl.user_session.get("react_graph")
    thread_config = cl.user_session.get("config")  # Retrieve stored configuration

    # Wrap the incoming user message as a HumanMessage
    messages = [HumanMessage(content=message.content)]

    # Create a RunnableConfig using the stored configurable keys and callback handlers
    runnable_config = RunnableConfig(
        callbacks=[
            cl.LangchainCallbackHandler(
                to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]
            )
        ],
        configurable=thread_config["configurable"]
    )

    # Invoke the graph with messages and our configured RunnableConfig
    response = react_graph.invoke({"messages": messages}, runnable_config)

    # Collect responses
    responses = []
    for msg in response.get('messages', []):
        content = getattr(msg, "content", str(msg)) if hasattr(msg, "content") else str(msg)
        if content:
            responses.append(content)

    # Send final response to the user
    if responses:
        await cl.Message(content=responses[-1]).send()
