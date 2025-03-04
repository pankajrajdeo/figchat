# generate_image_description_tool.py
import os
import json
import base64
import threading
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from preload_datasets import TRAIN_IMAGE_DATA_FILE

# Load environment variables
load_dotenv()

BASE_URL = "https://devapp.lungmap.net"  # <-- BASE URL for public image paths
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")

# Initialize a lock for thread-safe file operations
log_lock = threading.Lock()

def load_image_log() -> dict:
    """
    Load the existing log from TRAIN_IMAGE_DATA_FILE.
    If the file doesn't exist, initialize with an empty structure.
    """
    if not os.path.exists(TRAIN_IMAGE_DATA_FILE):
        return {"image_descriptions": []}

    try:
        with open(TRAIN_IMAGE_DATA_FILE, "r", encoding="utf-8", errors="replace") as f:
            return json.load(f)
    except json.JSONDecodeError:
        # If the file is corrupted, reset it
        return {"image_descriptions": []}

def append_image_log(image_path: str, query: str, description: str) -> None:
    """
    Append a new image description entry to the TRAIN_IMAGE_DATA_FILE.
    
    Parameters:
    - image_path: Path to the image.
    - query: The user-provided query.
    - description: The AI-generated image description.
    """
    with log_lock:  # Ensure thread-safe access
        log_data = load_image_log()
        log_entry = {
            "image_path": image_path,
            "query": query,
            "description": description,
            "timestamp": pd.Timestamp.now().isoformat()
        }
        log_data["image_descriptions"].append(log_entry)
        with open(TRAIN_IMAGE_DATA_FILE, "w") as f:
            json.dump(log_data, f, indent=4)

def Image_Analyzer(image_path: str, query: str) -> dict:
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
    with open(relative_path, "rb") as img_file:
        image_data = base64.b64encode(img_file.read()).decode("utf-8")

    # Construct prompt with instructions for concise responses
    enhanced_query = f"""Analyze this image and answer the following query concisely:

Query: {query}

IMPORTANT INSTRUCTIONS:
1. Avoid unnecessary repetition or verbose explanations
2. Focus on the specific aspects asked in the query
3. Present information in a structured format
4. Limit your response to essential information only
5. Do NOT repeat the query or restate what the image represents"""

    # Construct LLM message with user query and image
    message = HumanMessage(
        content=[
            {"type": "text", "text": enhanced_query},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_data}"}},
        ]
    )

    # Initialize and invoke the Google Gemini model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    
    response = model.invoke([message])
    description = response.content.strip()

    # If description is empty or invalid, do not log it
    if not description:
        return {"error": "Failed to generate image description.", "image_path": relative_path, "query": query}

    # Append to log file
    append_image_log(relative_path, query, description)

    # Return structured response
    return {
        "image_path": relative_path,
        "query": query,
        "description": description,
        "timestamp": pd.Timestamp.now().isoformat()
    }

# For backward compatibility
generate_image_description_tool = Image_Analyzer
