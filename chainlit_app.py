import os
import warnings
import uuid
from dotenv import load_dotenv
import pandas as pd
import chainlit as cl
import sqlite3

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
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

BASE_DATASET_DIR = os.getenv("BASE_DATASET_DIR")
PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")
DATASET_INDEX_FILE = os.getenv("DATASET_INDEX_FILE")
DATABASE_PATH = os.getenv("DATABASE_PATH")

# Preload datasets
PRELOADED_DATA = {}
PRELOADED_DATASET_INDEX = None

# Database connection
conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)

def preload_dataset_index(file_path: str):
    global PRELOADED_DATASET_INDEX
    try:
        PRELOADED_DATASET_INDEX = pd.read_csv(file_path, sep='\t')
    except Exception as e:
        print(f"Failed to preload dataset index: {e}")

def preload_all_h5ad_files(base_dir: str):
    global PRELOADED_DATA
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            if file.endswith(".h5ad"):
                fullpath = os.path.join(subdir_path, file)
                if fullpath not in PRELOADED_DATA:
                    try:
                        import scanpy as sc  # Import only when necessary
                        PRELOADED_DATA[fullpath] = sc.read_h5ad(fullpath)
                    except Exception as e:
                        print(f"Failed to load {fullpath}: {e}")

preload_dataset_index(DATASET_INDEX_FILE)
preload_all_h5ad_files(BASE_DATASET_DIR)

# Import and configure tools
import visualization_tool
import dataset_info_tool
import internet_search_tool

visualization_tool.PRELOADED_DATA = PRELOADED_DATA
visualization_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX
visualization_tool.PLOT_OUTPUT_DIR = PLOT_OUTPUT_DIR
dataset_info_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX

# Define LLM and Tools
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
tools = [visualization_tool.visualization_tool, dataset_info_tool.dataset_info_tool, internet_search_tool.internet_search_tool]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

sys_msg = SystemMessage(content="""You are LungMAP scExplore, an assistant to explore and visualize scRNA-seq Datasets from LungMAP. You specialize in generating UMAP Cluster plots and providing information about the LungMAP datasets you have access to. Currently, you can generate UMAP Cluster plots for the following four datasets:

1. **Human Lung Cell Atlas (HLCA) Metacells**:
   - Description: A large-scale integrated single-cell atlas of the human lung, reducing 2.28M cells to 50K metacells.
   - Source: LungMAP (Human Lung Cell Atlas Initiative).
   - File Name: `HLCA_full_superadata_v3_norm_log_deg.h5ad`

2. **Fetal Lung Development (HCA)**:
   - Description: Multiomic atlas of human lung development during 5–22 post-conception weeks, revealing developmental-specific cell states.
   - Source: LungMAP (Wellcome HCA Strategic Science Support).
   - File Name: `HCA_fetal_lung_normalized_log_deg.h5ad`

3. **Infant BPD Study (Sun Lab)**:
   - Description: Study of bronchopulmonary dysplasia (BPD) in infants using single-nucleus RNA sequencing, identifying alveolar dysplasia.
   - Source: LungMAP (Sun lab).
   - File Name: `BPD_infant_Sun_normalized_log_deg.h5ad`

4. **Fetal BPD Study (Sucre Lab)**:
   - Description: Analysis of preterm infant lungs and molecular dynamics driving bronchopulmonary dysplasia (BPD) and pulmonary hypertension (PH).
   - Source: LungMAP (Sucre lab).
   - File Name: `BPD_fetal_normalized_log_deg.h5ad`

### Tools Available:
- **visualization_tool**:
  - Use this tool to generate UMAP Cluster plots based on the user’s query. You can specify the dataset and choose the observation column for coloring the plot, (e.g. "cell_type" or "disease"). If the specified column is not available for the chosen dataset, inform the user and offer the option to generate the UMAP colored by another available column.
- **dataset_info_tool**:
  - Use this tool to provide detailed metadata and information about the datasets listed above.
- **internet_search_tool**:
  - Use this tool to perform an internet search for general queries that go beyond the preloaded dataset capabilities.

### Future Capabilities:
In future releases, you will be able to generate the following plot types:
- Heatmap
- Radar Plot
- Cell Frequency Boxplot
- Volcano Plot
- Dot Plot
- Violin Plot
- UMAP Gene Expression
- Venn Diagram
- UpSet Plot
- Gene Interaction Network

### Guidelines for Use:
- If the user query involves generating a UMAP Cluster plot, you should use the **visualization_tool**.
- If the user asks for information about a dataset, you should use the **dataset_info_tool**.
- If the query cannot be answered using the above tools or relates to external knowledge, use the **internet_search_tool**.

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
    await cl.Message(content="Welcome to LungMAP scExplore! How can I assist you today?").send()

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
        
