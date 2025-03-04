# app.py
import os
import warnings
import uuid
from dotenv import load_dotenv
import chainlit as cl
import asyncio
from typing import AsyncIterator, Dict, Any, Optional, TypedDict
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
            message="Please display side-by-side UMAPs of each clinical condition in the Sucre Lab BPD dataset, colored by the cell types that are present in each.",
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

# Pass the preloaded data to the tools
visualization_tool.PRELOADED_DATA = PRELOADED_DATA
visualization_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX
visualization_tool.PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")
dataset_info_tool.PRELOADED_DATASET_INDEX = PRELOADED_DATASET_INDEX

# Define LLM and Tools
llm = ChatOpenAI(
    model="gpt-4o-mini-2024-07-18",
    streaming=True  # Need to keep streaming for proper Chainlit integration
)

# Define the list of tools (without wrapping)
tools = [
    visualization_tool.visualization_tool,
    dataset_info_tool.dataset_info_tool,
    internet_search_tool.internet_search_tool,
    generate_image_description_tool.generate_image_description_tool
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
- **visualization_tool**: Use this tool to generate visualizations based on the user's query. You can specify the dataset and choose the observation column for customizing the plot (e.g., "cell_type", "disease", or other metadata fields).
- **dataset_info_tool**: Use this tool to provide detailed metadata and information about the datasets. **Note:** You can either:
  - Retrieve the preloaded metadata (default route) which shows all dataset details, or 
  - Provide a TSV file path in your query to parse and display the contents of that file.
- **internet_search_tool**: Use this tool to perform an internet search for general queries that go beyond the preloaded dataset capabilities.
- **generate_image_description_tool:** Use this tool to generate a comprehensive, detailed description of an uploaded image, analyzing its textual and visual elements.

### Tool Usage Guidelines:
- **Always explain your intent and specify relevant details before calling a tool**. For example:
  - "To answer your question about cell types in the Fetal BPD Study (Sucre Lab) dataset, I’ll check the available clinical conditions and cell types using the dataset_info_tool..."
  - "To visualize the gene expression patterns you’re asking about, I’ll generate a dot plot using the visualization_tool with the AT2 marker genes from the HCA_fetal_lung_normalized_log_deg.h5ad dataset..."
- **For visualization requests**: When the user requests a plot (e.g., heatmap, UMAP, gene regulatory network), **automatically call the visualization_tool** with the specified dataset and parameters unless the user explicitly instructs otherwise. Do not assume or fabricate plot generation or file paths without calling the tool.
- **Prevent hallucination**: Never claim a plot has been generated or provide fictitious file paths unless the visualization_tool has been invoked and returned actual results. If unsure about the user’s intent (e.g., whether they want a plot), ask for confirmation (e.g., "Would you like me to generate a heatmap for this data?") rather than assuming or hallucinating.

### Response Formatting:
- **IMPORTANT**: After calling a tool, **always insert a clear line break or start a new paragraph** before presenting the results. For example:
  - "Let me retrieve the dataset information now... [line break] The metadata shows..."
  - "Let me generate the heatmap now... [line break] I have generated the heatmap..."
- Ensure there is a visible separation between your preparatory steps and the tool’s output in your response.

### Handling LungMAP Queries:
- If a request is related to LungMAP.net or its resources, automatically construct the search URL as follows:
  - https://www.lungmap.net/search/?queries[]=$String
  - Replace $String with the user's search term and return the appropriate URL.

### Key Guidelines:
- **Using dataset_info_tool**:
  - **For Dataset Exploration**: Use this tool first when users ask about available datasets, cell types, or metadata. Mention the specific dataset and fields (e.g., cell types, clinical conditions) you're checking before invoking the tool.
  - **For TSV Analysis**: Use this tool when users ask about:
    - Key hub regulators in gene networks
    - Differentially expressed genes (DEGs) from heatmaps
    - Gene regulatory networks or interactions
    - Specific patterns in plots that generate TSV files (heatmaps, networks, volcano plots)
  - **Example triggers for TSV analysis**:
    - "Find key hub regulators in this network"
    - "What are the top DEGs in this heatmap?"
    - "Analyze the gene interactions in this network"

- **Using visualization_tool**:
  - Automatically invoke this tool when the user requests a visualization (e.g., "generate a gene regulatory network" or "make a heatmap"). Specify the dataset and parameters (e.g., cell types, genes) before calling it.
  - Example: "To generate a gene regulatory network from DEGs in abCAP cells from the BPD_fetal_normalized_log_deg.h5ad dataset, I’ll use the visualization_tool... next line break ...[results]"

- **Cross Check the Metadata**: If you are unsure whether a specific cell type, disease, or any other field is available for a user query, first consult the dataset_info_tool to explore the metadata. Explicitly state the dataset and fields you’re verifying.

- **Heatmap, Dotplot, and Stats Plots**: These work even if the user does not provide a gene list, as they are precomputed internally. Do not ask the user for gene symbols unless they are explicitly provided.

- **Using generate_image_description_tool**:  
  - If a user asks about patterns, trends, or notable features in a generated plot, use the **generate_image_description_tool** instead of inferring based on general knowledge. Mention that you’re analyzing the specific plot generated earlier.
  - Example triggers:  
    - "What is prominent in this image?"  
    - "What insights can you derive from this?"  
    - "Describe the patterns you see in this plot."

Always strive to understand the user's intent and provide accurate, context-appropriate responses based on the tools and datasets at your disposal. Mention specific datasets, cell types, or fields before invoking tools to build user trust and clarity. Never hallucinate plot generation or file paths—either call the visualization_tool directly or ask the user for clarification."""
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

    try:
        # Create a streaming message for the main assistant response
        ui_message = cl.Message(content="")
        await ui_message.send()
        logger.debug("Started new UI message")
        
        # For internal tracking of tool execution (not shown to user)
        current_tool_outputs = {}
        tool_steps = {}
        chain_sequence = []  # Track chain sequence
        current_chain_depth = 0  # Track chain nesting
        
        # Track if we're currently in a visualization tool execution
        in_visualization_tool = False
        visualization_tool_name = "Data_Visualizer"
        
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
                # Only stream tokens from the main assistant, not from internal LLM chains
                # Check if we're in a visualization tool execution
                if in_visualization_tool:
                    # For visualization tool, we'll only log but not stream to UI
                    content = event["data"]["chunk"].content
                    if content:
                        logger.debug(f"Visualization tool LLM output (not streamed): {content[:50]}...")
                else:
                    # For the main assistant, stream to UI as normal
                    content = event["data"]["chunk"].content
                    if content:
                        await ui_message.stream_token(content)
                        logger.debug(f"Streamed token: {content[:20]}...")
            
            elif event["event"] == "on_tool_start":
                # Log when a tool starts execution (internal only)
                tool_name = event["name"]
                logger.info(f"\nTool Execution Started: {tool_name}")
                logger.info("Chain sequence up to tool start:")
                for depth, seq in chain_sequence:
                    indent = "  " * depth
                    logger.info(f"{indent}{seq}")
                
                current_tool_outputs[tool_name] = ""
                
                # Check if this is the visualization tool
                if tool_name == visualization_tool_name:
                    in_visualization_tool = True
                    logger.info("Entered visualization tool execution - suppressing internal LLM streaming")
                
                # Create a step but don't display it in the main UI flow
                step = cl.Step(name=tool_name, type="tool", show_input=False)
                await step.__aenter__()
                tool_steps[tool_name] = step
                
            elif event["event"] == "on_tool_end":
                # Handle tool output (log but don't display in UI)
                tool_name = event["name"]
                tool_output = str(event["data"]["output"])
                if tool_output:
                    # Store the tool output but don't stream to UI
                    current_tool_outputs[tool_name] = tool_output
                    logger.info(f"\nTool {tool_name} completed")
                    logger.info("Chain sequence during tool execution:")
                    for depth, seq in chain_sequence:
                        indent = "  " * depth
                        logger.info(f"{indent}{seq}")
                    
                    # Log the tool output for debugging
                    logger.debug(f"Tool output for {tool_name}: {tool_output[:100]}...")
                    
                    # Update the step with the output but then remove it to keep UI clean
                    if tool_name in tool_steps:
                        step = tool_steps[tool_name]
                        step.output = tool_output
                        await step.__aexit__(None, None, None)
                        await step.remove()  # Remove the step to keep the UI clean
                
                # Check if we're exiting the visualization tool
                if tool_name == visualization_tool_name:
                    in_visualization_tool = False
                    logger.info("Exited visualization tool execution - resuming normal streaming")
            
            elif event["event"] == "on_chain_end" and event["name"] == "LangGraph":
                # Update state when the graph execution completes
                state = event["data"]["output"]
                cl.user_session.set("state", state)
                logger.debug("Graph execution completed")
                logger.info("\nFinal chain sequence:")
                for depth, seq in chain_sequence:
                    indent = "  " * depth
                    logger.info(f"{indent}{seq}")

        # Update the final message
        await ui_message.update()
        
        # Get the final content for chat history
        final_content = ui_message.content
        chat_history.append({"role": "assistant", "content": final_content})
        cl.user_session.set("chat_history", chat_history)
        logger.debug(f"Updated chat history with AI response. Total messages: {len(chat_history)}")
        
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
