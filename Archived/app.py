# app.py
import os
import warnings
import uuid
from dotenv import load_dotenv
import sqlite3
import chainlit as cl
import asyncio  # Added for async usage

from preload_datasets import PRELOADED_DATA, PRELOADED_DATASET_INDEX, DATABASE_PATH
from utils import parse_tsv_data
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
# BEFORE:
# from langgraph.checkpoint.sqlite import SqliteSaver
# AFTER:
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Import and configure tools
import visualization_tool
import dataset_info_tool
import internet_search_tool
import generate_image_description_tool

# Pass the preloaded data to the tools
visualization_tool.PRELOADED_DATA = PRELOADED_DATA
visualization_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX
visualization_tool.PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")
dataset_info_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX

# Define LLM and Tools
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
tools = [
    visualization_tool.visualization_tool,
    dataset_info_tool.dataset_info_tool,
    internet_search_tool.internet_search_tool,
    generate_image_description_tool.generate_image_description_tool
]
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

sys_msg = SystemMessage(content="""You are **LungMAP scExplore**, an advanced assistant for exploring and visualizing scRNA-seq Datasets from LungMAP. You specialize in generating multiple types of visualizations for lung-related single-cell datasets.

### Available Datasets:
1. **Human Lung Cell Atlas (HLCA) Metacells**:
   - A comprehensive atlas of 50K metacells
   - Source: Human Lung Cell Atlas Initiative
   - File: HLCA_full_superadata_v3_norm_log_deg.h5ad

2. **Fetal Lung Development (HCA)**:
   - Multiomic atlas covering 5–22 post-conception weeks
   - Source: Wellcome HCA Strategic Science Support
   - File: HCA_fetal_lung_normalized_log_deg.h5ad

3. **Infant BPD Study (Sun Lab)**:
   - Investigates bronchopulmonary dysplasia (BPD) in infants
   - Source: LungMAP (Sun lab)
   - File: BPD_infant_Sun_normalized_log_deg.h5ad

4. **Fetal BPD Study (Sucre Lab)**:
   - Explores molecular dynamics of BPD and pulmonary hypertension in preterm infants
   - Source: LungMAP (Sucre lab)
   - File: BPD_fetal_normalized_log_deg.h5ad

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
- **visualization_tool**: Use this tool to generate visualizations based on the user's query. You can specify the dataset and choose the observation column for customizing the plot (e.g., "cell_type", "disease", or other metadata fields).
- **dataset_info_tool**: Use this tool to provide detailed metadata and information about the datasets. **Note:** You can either:
  - Retrieve the preloaded metadata (default route) which shows all dataset details, or 
  - Provide a TSV file path in your query to parse and display the contents of that file.
- **internet_search_tool**: Use this tool to perform an internet search for general queries that go beyond the preloaded dataset capabilities.
- **generate_image_description_tool:** Use this tool to generate a comprehensive, detailed description of an uploaded image, analyzing its textual and visual elements.

### Handling LungMAP Queries:
- If a request is related to LungMAP.net or its resources, automatically construct the search URL as follows:
  - https://www.lungmap.net/search/?queries[]=$String
  - Replace $String with the user's search term and return the appropriate URL.
                        
### Key Guidelines:
- **Cross Check the Metadata**: If you are unsure whether a specific cell type, disease, or any other field is available for a user query, first consult the Dataset Information Tool to explore the metadata.
- **Heatmap, Dotplot, and Stats Plots**: These work even if the user does not provide a gene list, as they are precomputed internally. Do not ask the user for gene symbols unless they are explicitly provided.
- **Always Use generate_image_description_tool for Plot Interpretation**:  
  - If a user asks about patterns, trends, or notable features in a generated plot, use the **generate_image_description_tool** instead of inferring based on general knowledge.  
  - Example triggers:  
    - "What is prominent in this image?"  
    - "What insights can you derive from this?"  
    - "Describe the patterns you see in this plot."
    
Always strive to understand the user's intent and provide accurate and context-appropriate responses based on the tools and datasets at your disposal."""
)

# --- Updated assistant node function ---
async def assistant(state: MessagesState):
    # Prepend the system message to the conversation history and await the LLM response
    response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# Chainlit Handlers
@cl.on_chat_start
async def on_chat_start():
    """Initialize session-specific settings when a new chat begins."""
    # Create a unique checkpoint namespace for this session
    thread_id = str(uuid.uuid4())
    checkpoint_ns = f"session_{thread_id}"
    
    # Build the graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")
    
    try:
        # Create a new saver using the DATABASE_PATH from environment
        if not DATABASE_PATH:
            raise ValueError("DATABASE_PATH not set in environment variables")
            
        saver = AsyncSqliteSaver.from_conn_string(DATABASE_PATH)
        # Store the context manager in the session
        cl.user_session.set("saver_cm", saver)
        # Enter the context
        saver_instance = await saver.__aenter__()
        # Initialize the saver
        await saver_instance.setup()
        
        # Compile the graph with the saver
        react_graph = builder.compile(checkpointer=saver_instance)
        
        # Save the compiled graph and config in the user session
        config = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": ""
            }
        }
        cl.user_session.set("react_graph", react_graph)
        cl.user_session.set("config", config)
        cl.user_session.set("saver", saver_instance)
        
        await cl.Message(
            content=(
                "Welcome to **LungMAP scExplore**, an agent-based AI framework to generate advanced single-cell genomics data visualizations through conversation. How can I assist you today?\n\n"
                "Here are some example queries to get you started:\n\n"
                "1. Please list the datasets, sampled tissues, and clinical conditions that you are able to allow me to explore.\n"
                "2. How can I compare clinical conditions that are sampled in these datasets?\n"
                "3. What data visualizations are you able to carry out for any of these conditions and their corresponding cell types?\n"
                "4. Please display side-by-side UMAPs of each clinical condition in the Sucre Lab BPD dataset, colored by the cell types that are present in each.\n"
                "5. Please display side-by-side UMAPs of the Sucre et al. clinical conditions, colored by EDNRB.\n"
                "6. Please show heatmaps of the differentially expressed genes (DEGs) in abCAP cells from the BPD+PHT samples.\n"
                "7. Please generate a gene regulatory network from the DEGs in abCAP cells from the BPD+PHT samples.\n"
                "8. What is the significance of the downregulated genes in abCAP endothelial cells from the BPD+PHT samples?\n\n"
                "**Note:**\n"
                "1. Plot generation and detailed description generation may take a minute or two. Please wait while it is being generated.\n"
                "2. In case the session gets stuck, please click on the notepad icon in the upper corner."
            )
        ).send()
    except Exception as e:
        print(f"Error in chat start: {e}")
        await cl.Message(content=f"An error occurred while starting the chat: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages and provide responses."""
    react_graph = cl.user_session.get("react_graph")
    thread_config = cl.user_session.get("config")
    saver = cl.user_session.get("saver")

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

    try:
        # Asynchronously invoke the graph with messages and our configured RunnableConfig
        response = await react_graph.ainvoke({"messages": messages}, runnable_config)

        # Collect responses
        responses = []
        for msg in response.get('messages', []):
            content = getattr(msg, "content", str(msg)) if hasattr(msg, "content") else str(msg)
            if content:
                responses.append(content)

        # Send final response to the user
        if responses:
            await cl.Message(content=responses[-1]).send()
    except Exception as e:
        print(f"Error in message handling: {str(e)}")
        await cl.Message(content=f"An error occurred while processing your message: {str(e)}").send()

@cl.on_chat_end
async def on_chat_end():
    """Clean up resources when the chat ends."""
    try:
        saver_cm = cl.user_session.get("saver_cm")
        if saver_cm:
            await saver_cm.__aexit__(None, None, None)
    except Exception as e:
        print(f"Error during chat cleanup: {e}")
