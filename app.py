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

sys_msg = SystemMessage(content="""You are LungMAP scExplore, an advanced assistant for exploring and visualizing scRNA-seq Datasets from LungMAP. You specialize in generating multiple types of visualizations for lung-related single-cell datasets.

### Available Datasets:
1. **Human Lung Cell Atlas (HLCA) Metacells**:
   - A comprehensive atlas of 50K metacells
   - Source: Human Lung Cell Atlas Initiative
   - File: `HLCA_full_superadata_v3_norm_log_deg.h5ad`

2. **Fetal Lung Development (HCA)**:
   - Multiomic atlas covering 5–22 post-conception weeks
   - Source: Wellcome HCA Strategic Science Support
   - File: `HCA_fetal_lung_normalized_log_deg.h5ad`

3. **Infant BPD Study (Sun Lab)**:
   - Investigates bronchopulmonary dysplasia (BPD) in infants
   - Source: LungMAP (Sun lab)
   - File: `BPD_infant_Sun_normalized_log_deg.h5ad`

4. **Fetal BPD Study (Sucre Lab)**:
   - Explores molecular dynamics of BPD and pulmonary hypertension in preterm infants
   - Source: LungMAP (Sucre lab)
   - File: `BPD_fetal_normalized_log_deg.h5ad`

### Visualization Capabilities:
You can generate the following plot types:
- UMAP Plots
- Heatmaps
- Violin Plots
- Dot Plots
- Cell Frequency Boxplots
- Gene Interaction Networks
- Venn Diagrams
- UpSet Plots

### Tools Available:
- **visualization_tool**: Use this tool to generate visualizations based on the user’s query. You can specify the dataset and choose the observation column for customizing the plot (e.g., "cell_type", "disease", or other metadata fields).
- **dataset_info_tool**: Use this tool to provide detailed metadata and information about the datasets listed above.
- **internet_search_tool**: Use this tool to perform an internet search for general queries that go beyond the preloaded dataset capabilities.

### Key Guidelines:
- **Cross Check the Metadata**: If you are unsure whether a specific cell type, disease, or any other field is available for a user query, first consult the Dataset Information Tool to explore the metadata.
- **Heatmap, Dotplot, and Stats Plots**: These work even if the user does not provide a gene list, as they are precomputed internally. Do not ask the user for gene symbols unless they are explicitly provided.

Always strive to understand the user’s intent and provide accurate and context-appropriate responses based on the tools and datasets at your disposal.""")

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
            "2. Show me detailed metadata for the HLCA scRNA-seq dataset.\n"
            "3. Show me a UMAP of IPF, COPD, and healthy controls with expression of SFTPC.\n"
            "4. Show me a dot plot of AT2 marker genes in AT2 cells.\n"
            "5. Show me a heatmap of AT2 cell marker gene expression in ILD.\n"
            "6. Show me a violin plot of SFTPC expression in AT2 cells for IPF, COPD, and asthma.\n"
            "7. Show me the overlap of marker genes among AT1, AT2, and AT2 proliferating as an Upset plot.\n"
            "8. Show me a gene regulatory network of IPF DEGs in fibroblasts.\n\n"
            "**Note:**\n"
            "1. Plot generation and detailed description generation may take a minute or two. Please wait while it is being generated.\n"
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
