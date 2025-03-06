# app.py
import os
import warnings
import uuid
from dotenv import load_dotenv
import chainlit as cl
import asyncio
import pandas as pd
import threading
from typing import AsyncIterator, Dict, Any, Optional, TypedDict, List
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from sqlalchemy import event
from sqlalchemy.engine import Engine
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
import json
import ast
from preload_datasets import PRELOADED_DATA, PRELOADED_DATASET_INDEX, DATABASE_URL
from utils import parse_tsv_data
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, START, StateGraph
from langgraph.prebuilt import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
import logging

# Set up logging
logger = logging.getLogger("lungmap_lungchat")
logging_level = os.getenv('LOGGING_LEVEL', 'INFO').upper()
numeric_level = getattr(logging, logging_level, None)
logger.setLevel(numeric_level)

# --------------------------
# SFT Training Data Collection
# --------------------------
# Path for storing SFT training data
SFT_TRAINING_DATA_FILE = os.path.join(os.environ.get("BASE_DATASET_DIR", ""), "training_data", "sft_training_data.json")

# Initialize a lock for thread-safe file operations
log_lock = threading.Lock()

def load_sft_log() -> dict:
    """
    Load the existing SFT training log from SFT_TRAINING_DATA_FILE.
    If the file doesn't exist, initialize with an empty structure.
    """
    os.makedirs(os.path.dirname(SFT_TRAINING_DATA_FILE), exist_ok=True)
    if not os.path.exists(SFT_TRAINING_DATA_FILE):
        return {"interactions": []}

    try:
        with open(SFT_TRAINING_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted, reset it
        return {"interactions": []}

def append_sft_log(
    user_message: str, 
    assistant_response: str, 
    tool_calls: List[Dict], 
    interaction_type: str
) -> None:
    """
    Append a new SFT training entry to the log file.
    
    Parameters:
    - user_message: The user query/message
    - assistant_response: The final response from the assistant
    - tool_calls: List of tool calls made during the interaction
    - interaction_type: Type of interaction ("direct_tool_call", "explanatory_tool_call", or "direct_response")
    """
    with log_lock:  # Ensure thread-safe access
        log_data = load_sft_log()
        log_entry = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "tool_calls": tool_calls,
            "interaction_type": interaction_type,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        log_data["interactions"].append(log_entry)
        with open(SFT_TRAINING_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=4)

# PostgreSQL connection configuration
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

@cl.data_layer
def get_data_layer():
    """Initialize the PostgreSQL data layer for Chainlit."""
    return SQLAlchemyDataLayer(conninfo=DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://"))

# Add header authentication
@cl.header_auth_callback
def header_auth_callback(headers: Dict) -> Optional[cl.User]:
    """
    Authenticate users based on request headers.
    Required header:
    - Shib-Eduperson-Principal-Name: The Shibboleth user identifier
    """
    user_id = headers.get("Shib-Eduperson-Principal-Name")
    
    if user_id is not None:
        return cl.User(
            identifier=user_id,
            metadata={
                "role": "admin",
                "provider": "header"
            }
        )
    return cl.User(
            identifier="default_user",
            metadata={
                "role": "user",
                "provider": "default"
            }
        )

# Define starters for the application
@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Explore Available Datasets",
            message="Please list the datasets, sampled tissues, and clinical conditions that you are able to allow me to explore.",
            icon="public/data-analysis-svgrepo-com.svg"
        ),
        cl.Starter(
            label="UMAP Visualization",
            message="Please display side-by-side UMAPs of each clinical condition in the Sucre Lab BPD dataset, colored by the cell types that are present in each condition.",
            icon="public/scatter-plot-svgrepo-com.svg"
        ),
        cl.Starter(
            label="Gene Expression Heatmap",
            message="Show me a heatmap of AT2 cell marker gene expression in ILD.",
            icon="public/heatmap-svgrepo-com.svg"
        ),
        cl.Starter(
            label="Gene Regulatory Network",
            message="Please generate a gene regulatory network from the differentially expressed genes in abCAP cells from the BPD+PHT samples.",
            icon="public/network-graph-presentation-svgrepo-com.svg"
        ),
        cl.Starter(
            label="Marker Gene Overlap",
            message="Show me the overlap of marker genes among AT1, AT2, and AT2 proliferating as a Venn diagram.",
            icon="public/venn-diagram-svgrepo-com.svg"
        ),
        cl.Starter(
            label="Dot Plot of Marker Genes",
            message="Show me a dot plot of AT2 marker genes in AT2 cells.",
            icon="public/bubble-chart-svgrepo-com.svg"
        ),
    ]

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()

# Import and configure tools
import visualization_tool
import dataset_info_tool
import internet_search_tool
import generate_image_description_tool
import code_generation_tool

# Pass the preloaded data to the tools
visualization_tool.PRELOADED_DATA = PRELOADED_DATA
visualization_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX
visualization_tool.PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")
dataset_info_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX
code_generation_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX
code_generation_tool.PRELOADED_DATA = PRELOADED_DATA
code_generation_tool.PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")
code_generation_tool.BASE_DATASET_DIR = os.getenv("BASE_DATASET_DIR")

# Define LLM and Tools
llm = ChatOpenAI(
    model="gpt-4o",
    streaming=True  # Need to keep streaming for proper Chainlit integration
)

# Define the list of tools (without wrapping)
tools = [
    visualization_tool.visualization_tool,
    dataset_info_tool.dataset_info_tool,
    internet_search_tool.internet_search_tool,
    generate_image_description_tool.generate_image_description_tool,
    code_generation_tool.code_generation_tool
]

# Bind the tools to the LLM
llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

sys_msg = SystemMessage(content="""You are **LungChat**, an advanced assistant for exploring and visualizing scRNA-seq Datasets from LungMAP. You specialize in generating multiple types of visualizations for lung-related single-cell datasets.

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
- **visualization_tool**: Use this tool to generate visualizations based on the user's query. The tool will automatically select the most appropriate dataset based on the query content. You can customize the plot using observation columns (e.g., "cell_type", "disease", or other metadata fields).
- **dataset_info_tool**: Use this tool to provide detailed metadata and information about the datasets. **Note:** You can either:
  - Retrieve the preloaded metadata (default route) which shows all dataset details, or
  - Provide a TSV file path in your query to parse and display the contents of that file.
- **internet_search_tool**: Use this tool to perform an internet search for general queries that go beyond the preloaded dataset capabilities.
- **generate_image_description_tool**: Use this tool to generate a comprehensive, detailed description of an uploaded image, analyzing its textual and visual elements.
- **code_generation_tool**: Use this tool to generate and execute custom Python code for complex analyses that are not covered by the standard visualization and dataset tools. This tool is particularly useful for advanced users who need to perform specific data manipulations or analyses that require custom code execution.

### Tool Usage Guidelines:
- **Always explain your intent and specify relevant details before calling a tool**. For example:
  - "To answer your question about cell types in the Fetal BPD Study (Sucre Lab) dataset, I'll check the available clinical conditions and cell types..."
  - "To visualize the gene expression patterns you're asking about, I'll generate a dot plot with the AT2 marker genes..."
- **For visualization requests**: When the user requests a plot (e.g., heatmap, UMAP, gene regulatory network), **call the visualization_tool** with the appropriate parameters unless the user explicitly instructs otherwise. The tool will select the most relevant dataset based on the query. For each generated visualization, ALWAYS show ALL PNG images directly in your response using Markdown image syntax `![Description](image_url)`. If multiple PNG images are generated, display ALL of them. For PDFs, TSVs, and other non-image outputs, provide clickable links to ALL generated files.
- **For custom code generation**: When the user requests a specific analysis or data manipulation that requires custom code, **call the code_generation_tool** to generate and execute the necessary Python code.
- **Never mention the tool name in your response**.
- **Reusing previous tool outputs**: When a user asks to see the code or configuration used to generate a previous output:
  - For **code_generation_tool**: Present the code that was used (or failed code with error) from the previous tool call output without invoking the tool again. Format as: "Here is your requested plot\nFollowing code was used to generate this plot: [code]"
  - For **visualization_tool**: Present the configuration used to generate the files from the previous tool call output without invoking the tool again. Format as: "Here is your requested plot\nFollowing config was used to generate this plot: [config]"

### IMPORTANT:
- **Do not hallucinate**: Never claim a plot has been generated or provide fictitious file paths unless the visualization_tool has been invoked and returned actual results. Under no circumstances hallucinate plot generation or file paths; always invoke the visualization_tool or ask the user for clarification first.
- **Mandatory Tool Invocation**: You must always call the appropriate tool before providing any specific details or confirmations; fabricating or assuming outputs without tool invocation is strictly prohibited—if uncertain, you must ask the user for clarification (e.g., "Would you like me to generate a heatmap for this data?").
- **Image Display**: When visualization_tool generates plots, ALWAYS display ALL PNG images inline using Markdown image syntax: `![Description](image_url)` instead of just providing download links. If multiple PNG images are generated for a single plot type, display ALL of them in your response. Specifically, for EACH png_path returned by the tool, include the full image in your response using Markdown image syntax. For other output types (PDF files, TSV files, etc.), provide clickable links to ALL of them in your response.
- **Response Formatting**: After calling a tool, always insert a clear line break or start a new paragraph before presenting the results. For example:
  - "Let me retrieve the dataset information now... \n\n The metadata shows..."
  - "Let me generate the heatmap now... \n\n I have generated the heatmap..."
  - "Let me generate the code now... \n\n I have generated the code..."
  - "Let me create UMAP visualizations... \n\n Here are the UMAPs showing cell clusters: \n\n ![UMAP Visualization 1](https://example.com/plot1.png) \n\n ![UMAP Visualization 2](https://example.com/plot2.png) \n\n You can also download the PDF versions ([PDF 1](https://example.com/plot1.pdf), [PDF 2](https://example.com/plot2.pdf)) or access the raw data ([TSV 1](https://example.com/data1.tsv), [TSV 2](https://example.com/data2.tsv)) for further analysis."

### Handling LungMAP Queries:
- If a request is related to LungMAP.net or its resources, automatically construct the search URL as follows:
  - https://www.lungmap.net/search/?queries[]=$String
  - Replace $String with the user's search term and return the appropriate URL.

### Key Guidelines:
- **Dataset Exploration**: When users ask about available datasets, cell types, or metadata, use the dataset_info_tool first. Mention the specific dataset and fields (e.g., cell types, clinical conditions) you're checking.
- **TSV Analysis**: Use the dataset_info_tool when users ask about:
  - Key hub regulators in gene networks
  - Differentially expressed genes (DEGs) from heatmaps
  - Gene regulatory networks or interactions
  - Specific patterns in plots that generate TSV files (e.g., heatmaps, networks, volcano plots)
  - Example triggers: "Find key hub regulators in this network," "What are the top DEGs in this heatmap?"
- **Visualization Requests**: Invoke the visualization_tool for requests like "generate a gene regulatory network" or "make a heatmap." Specify parameters (e.g., cell types, genes) if provided; otherwise, the tool selects the dataset. For each generated visualization, ALWAYS show ALL PNG images directly in your response using Markdown image syntax `![Description](image_url)`. If multiple images are generated for a single plot type, display ALL of them. Additionally, include clickable links for ALL PDF files, TSV files, and other non-image outputs to give users easy access to all generated resources.
- **Metadata Verification**: If unsure about a specific cell type, disease, or field availability, consult the dataset_info_tool to explore the metadata, stating the dataset and fields you're verifying.
- **Precomputed Plots**: Heatmaps, dot plots, and stats plots work without a user-provided gene list (precomputed internally). Do not ask for gene symbols unless explicitly provided.
- **Image Analysis**: For questions about patterns, trends, or features in a generated plot (e.g., "What is prominent in this image?"), use the generate_image_description_tool, mentioning you're analyzing the specific plot.
- **Code Generation**: Use the code_generation_tool in the following cases:
  - When the user explicitly requests a custom TSV file (e.g., "Give me a custom TSV file with AT2 cell data"), custom data output (e.g., "Provide custom data for BPD samples"), or a custom plot not covered by visualization_tool (e.g., "Create a custom plot of gene expression trends over time").
  - As a fallback: If the visualization_tool, dataset_info_tool, or other tools cannot fulfill the user's request (e.g., the requested plot type or data manipulation isn't supported), attempt to use the code_generation_tool to generate custom code to meet the need, explaining: "The standard tools couldn't address this directly, so I'll generate custom code to handle your request. This may take a few minutes."
  - Provide the parameters for code generation as well as the dataset_name if specified; otherwise, it automatically selects the most relevant dataset.
Always strive to understand the user's intent and provide accurate, context-appropriate responses based on the tools and datasets at your disposal. Mention specific datasets, cell types, or fields before invoking tools to build user trust and clarity."""
)

# --- Updated assistant node function ---
async def assistant(state: MessagesState):
    # Prepend the system message to the conversation history and await the LLM response
    response = await llm_with_tools.ainvoke([sys_msg] + state["messages"])
    return {"messages": [response]}

# Add ThreadDict type definition
class ThreadDict(TypedDict):
    id: str
    metadata: Dict
    steps: list[Dict]

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
        # Create a connection pool for PostgreSQL
        pool = await AsyncConnectionPool(
            conninfo=DATABASE_URL,
            max_size=20,
            kwargs=connection_kwargs,
        ).__aenter__()
        
        # Create a new saver using the pool
        saver = AsyncPostgresSaver(pool)
        # Store the context manager in the session
        cl.user_session.set("saver_cm", pool)
        # Initialize the saver
        await saver.setup()
        
        # Compile the graph with the saver
        react_graph = builder.compile(checkpointer=saver)
        
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
        cl.user_session.set("saver", saver)
        cl.user_session.set("chat_history", [])
        
    except Exception as e:
        logger.error(f"Error in chat start: {e}")
        await cl.Message(content=f"An error occurred while starting the chat: {str(e)}").send()

@cl.on_message
async def on_message(message: cl.Message):
    """Handle user messages and provide responses using improved streaming."""
    # Log first 50 chars of message
    logger.info(f"Received message: {message.content[:50]}...")
    
    react_graph = cl.user_session.get("react_graph")
    thread_config = cl.user_session.get("config")
    chat_history = cl.user_session.get("chat_history", [])

    # Add user message to chat history
    chat_history.append({"role": "user", "content": message.content})
    cl.user_session.set("chat_history", chat_history)

    # Get all messages from chat history for context
    messages = []
    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(SystemMessage(content=msg["content"]))

    # Create a RunnableConfig using the stored configurable keys and callback handlers
    runnable_config = RunnableConfig(
        callbacks=[
            cl.LangchainCallbackHandler(
                stream_final_answer=True,
                to_ignore=["ChannelRead", "RunnableLambda", "ChannelWrite", "__start__", "_execute"]
            )
        ],
        configurable=thread_config["configurable"]
    )

    # Define tools that require suppression of internal LLM streaming
    suppress_streaming_tools = {"Data_Visualizer", "Code_Generator", "Dataset_Explorer", "Image_Analyzer", "Web_Search"}
    
    try:
        # Create a streaming message for the main assistant response
        ui_message = cl.Message(content="")
        await ui_message.send()
        logger.debug("Started new UI message")
        
        # For internal tracking of tool execution (not shown to user)
        current_tool_outputs = {}
        current_tool_inputs = {}  # Track tool inputs for SFT data collection
        tool_steps = {}
        chain_sequence = []  # Track chain sequence
        current_chain_depth = 0  # Track chain nesting
        
        # Track if we're currently in a tool execution that suppresses streaming
        in_suppress_streaming_tool = False
        current_suppress_tool = None
        
        # For SFT data collection
        streamed_before_tool = ""  # Track content streamed before first tool call
        has_tool_call = False  # Track if any tool was called
        collected_tool_calls = []  # Store all tool calls for SFT data
        
        # Stream the response using astream_events for more granular control
        logger.info("Starting to stream response")
        async for event in react_graph.astream_events(
            {"messages": messages}, 
            version="v1", 
            stream_mode="values",
            config=runnable_config
        ):
            # Handle different event types
            if event["event"] == "on_chain_start":
                current_chain_depth += 1
                chain_name = event.get("name", "unnamed_chain")
                indent = "  " * current_chain_depth
                logger.info(f"{indent}Chain Start: {chain_name}")
                chain_sequence.append((current_chain_depth, f"Started: {chain_name}"))

            elif event["event"] == "on_chain_end":
                chain_name = event.get("name", "unnamed_chain")
                indent = "  " * current_chain_depth
                logger.info(f"{indent}Chain End: {chain_name}")
                chain_sequence.append((current_chain_depth, f"Ended: {chain_name}"))
                current_chain_depth = max(0, current_chain_depth - 1)

            elif event["event"] == "on_chat_model_stream":
                if in_suppress_streaming_tool:
                    # For tools requiring suppression, log but don't stream to UI
                    content = event["data"]["chunk"].content
                    if content:
                        logger.debug(f"{current_suppress_tool} internal LLM output (not streamed): {content[:50]}...")
                else:
                    # For the main assistant, stream to UI as normal
                    content = event["data"]["chunk"].content
                    if content:
                        await ui_message.stream_token(content)
                        logger.debug(f"Streamed token: {content[:20]}...")
                        if not has_tool_call and content:
                            streamed_before_tool += content
            
            elif event["event"] == "on_tool_start":
                tool_name = event["name"]

                # Merge "input" and "kwargs" so we never lose the user query or the TSV path
                merged_input = {}
                if isinstance(event["data"].get("input"), dict):
                    merged_input.update(event["data"]["input"])
                if isinstance(event["data"].get("kwargs"), dict):
                    merged_input.update(event["data"]["kwargs"])
                tool_input_dict = merged_input
                tool_input_str = str(tool_input_dict)

                logger.info(f"\nTool Execution Started: {tool_name} | input: {tool_input_str}")
                logger.info("Chain sequence up to tool start:")
                for depth, seq in chain_sequence:
                    indent = "  " * depth
                    logger.info(f"{indent}{seq}")
                
                has_tool_call = True
                current_tool_outputs[tool_name] = ""
                current_tool_inputs[tool_name] = tool_input_str

                if tool_name in suppress_streaming_tools:
                    in_suppress_streaming_tool = True
                    current_suppress_tool = tool_name
                    logger.info(f"Entered {tool_name} execution - suppressing internal LLM streaming")

                step = cl.Step(name=tool_name, type="tool", show_input=False)
                await step.__aenter__()
                tool_steps[tool_name] = step
                
            elif event["event"] == "on_tool_end":
                tool_name = event["name"]
                tool_output = str(event["data"]["output"])
                if tool_output:
                    current_tool_outputs[tool_name] = tool_output
                    logger.info(f"\nTool {tool_name} completed")
                    logger.info("Chain sequence during tool execution:")
                    for depth, seq in chain_sequence:
                        indent = "  " * depth
                        logger.info(f"{indent}{seq}")
                    
                    logger.debug(f"Tool output for {tool_name}: {tool_output[:100]}...")
                    
                    collected_tool_calls.append({
                        "tool_name": tool_name,
                        "tool_input": current_tool_inputs.get(tool_name, ""),
                        "tool_output": tool_output,
                        "preamble": streamed_before_tool if streamed_before_tool else ""
                    })
                    
                    if tool_name in tool_steps:
                        step = tool_steps[tool_name]
                        step.output = tool_output
                        await step.__aexit__(None, None, None)
                        await step.remove()

                if tool_name in suppress_streaming_tools:
                    in_suppress_streaming_tool = False
                    current_suppress_tool = None
                    logger.info(f"Exited {tool_name} execution - resuming normal streaming")
            
            elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                state = event["data"]["output"]
                cl.user_session.set("state", state)
                logger.debug("Graph execution completed")
                logger.info("\nFinal chain sequence:")
                for depth, seq in chain_sequence:
                    indent = "  " * depth
                    logger.info(f"{indent}{seq}")

        await ui_message.update()
        
        final_content = ui_message.content
        chat_history.append({"role": "assistant", "content": final_content})
        cl.user_session.set("chat_history", chat_history)
        logger.debug(f"Updated chat history with AI response. Total messages: {len(chat_history)}")
        
        interaction_type = "direct_response"
        if has_tool_call:
            if streamed_before_tool and streamed_before_tool.strip():
                interaction_type = "explanatory_tool_call"
            else:
                interaction_type = "direct_tool_call"
        
        append_sft_log(
            user_message=message.content,
            assistant_response=final_content,
            tool_calls=collected_tool_calls,
            interaction_type=interaction_type
        )
        logger.info(f"Collected SFT training data with interaction type: {interaction_type}")
        
    except Exception as e:
        logger.error(f"Error in message handling: {str(e)}")
        await cl.Message(content=f"An error occurred while processing your message: {str(e)}").send()

@cl.on_chat_end
async def on_chat_end():
    """Clean up resources when the chat ends."""
    try:
        pool = cl.user_session.get("saver_cm")
        if pool:
            await pool.__aexit__(None, None, None)
            logger.info("Successfully closed PostgreSQL connection pool")
    except Exception as e:
        logger.error(f"Error during chat cleanup: {e}")

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    cl.user_session.set("chat_history", [])

    # user_session = thread["metadata"]
    
    for message in thread["steps"]:
        if message["type"] == "user_message":
            cl.user_session.get("chat_history").append({"role": "user", "content": message["output"]})
        elif message["type"] == "assistant_message":
            cl.user_session.get("chat_history").append({"role": "assistant", "content": message["output"]})
    
    # Initialize session-specific settings
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
        # Create a connection pool for PostgreSQL
        pool = await AsyncConnectionPool(
            conninfo=DATABASE_URL,
            max_size=20,
            kwargs=connection_kwargs,
        ).__aenter__()
        
        # Create a new saver using the pool
        saver = AsyncPostgresSaver(pool)
        # Store the context manager in the session
        cl.user_session.set("saver_cm", pool)
        # Initialize the saver
        await saver.setup()
        
        # Compile the graph with the saver
        react_graph = builder.compile(checkpointer=saver)
        
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
        cl.user_session.set("saver", saver)
        logger.info(f"Chat resumed for thread: {thread['id']}")
        
    except Exception as e:
        logger.error(f"Error in chat resume: {e}")
        await cl.Message(content=f"An error occurred while resuming the chat: {str(e)}").send()
