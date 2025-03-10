import os
import json
import base64
import threading
import pandas as pd
import aiofiles
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from preload_datasets import TRAIN_IMAGE_DATA_FILE, BASE_URL

# BASE_URL = "https://devapp.lungmap.net"  # <-- BASE URL for public image paths

# Initialize a lock for thread-safe file operations
log_lock = threading.Lock()

async def load_image_log() -> dict:
    """
    Load the existing log from TRAIN_IMAGE_DATA_FILE.
    If the file doesn't exist, initialize with an empty structure.
    """
    if not os.path.exists(TRAIN_IMAGE_DATA_FILE):
        return {"image_descriptions": []}

    try:
        async with aiofiles.open(TRAIN_IMAGE_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            content = await f.read()
            return json.loads(content)
    except json.JSONDecodeError:
        # If the file is corrupted, reset it
        return {"image_descriptions": []}

async def append_image_log(image_path: str, query: str, description: str) -> None:
    """
    Append a new image description entry to the TRAIN_IMAGE_DATA_FILE.
    
    Parameters:
    - image_path: Path to the image.
    - query: The user-provided query.
    - description: The AI-generated image description.
    """
    async with aiofiles.open(TRAIN_IMAGE_DATA_FILE, "r+", encoding="utf-8") as f:
        content = await f.read()
        log_data = json.loads(content) if content else {"image_descriptions": []}
        
        log_entry = {
            "image_path": image_path,
            "query": query,
            "description": description,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        log_data["image_descriptions"].append(log_entry)
        
        await f.seek(0)
        await f.write(json.dumps(log_data, indent=4))
        await f.truncate()

async def generate_image_description_tool(image_path: str, query: str) -> dict:
    """
    Generates a detailed description of a given image based on a user-provided query and logs the result.

    Parameters:
    - image_path (str): Path to the image file.
    - query (str): User-provided prompt for image description.

    Returns:
    - dict: JSON-like structured output containing:
        - "image_path": The original image file path.
        - "query": The user query provided for image analysis.
        - "description": AI-generated description of the image.
        - "timestamp": When the description was generated.
    """
    # --- UPDATED BLOCK TO ENSURE LEADING SLASH ---
    if image_path.startswith(BASE_URL):
        relative_path = "/" + image_path[len(BASE_URL):].lstrip("/")
    else:
        relative_path = image_path

    if not os.path.exists(relative_path):
        return {"error": "Image file not found.", "image_path": relative_path, "query": query}

    # Convert image to Base64
    async with aiofiles.open(relative_path, "rb") as img_file:
        image_data = base64.b64encode(await img_file.read()).decode("utf-8")

    # Construct LLM message with user query
    message = HumanMessage(
        content=[
            {"type": "text", "text": query},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    )

    # Initialize and invoke the model asynchronously
    model = ChatOpenAI(model="gpt-4o-mini-2024-07-18")
    response = await model.ainvoke([message])
    description = response.content.strip()

    # If description is empty or invalid, do not log it
    if not description:
        return {"error": "Failed to generate image description.", "image_path": relative_path, "query": query}

    # Append to log file
    await append_image_log(relative_path, query, description)

    # Return structured response
    return {
        "image_path": relative_path,
        "query": query,
        "description": description,
        "timestamp": pd.Timestamp.now().isoformat()
    }
