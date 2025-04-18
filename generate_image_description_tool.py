# generate_image_description_tool.py
import os
import json
import base64
import threading
import pandas as pd
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

# Load environment variables
load_dotenv()

try:
    BASE_URL = os.environ["BASE_URL"]  # <-- BASE URL for public image paths
except KeyError:
    raise ValueError("BASE_URL environment variable is not set. Please set it in your .env file.")

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")

def Image_Analyzer(image_path: str, query: str) -> dict:
    """
    Generates a detailed description of a given image based on a user-provided query and logs the result.
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
2. Limit your response to essential information
3. Do NOT repeat the query or restate what the image represents"""

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
        temperature=0
    )
    
    response = model.invoke([message])
    description = response.content.strip()

    # If description is empty or invalid, do not log it
    if not description:
        return {"error": "Failed to generate image description.", "image_path": relative_path, "query": query}

    # Return structured response
    return {
        "image_path": relative_path,
        "query": query,
        "description": description,
        "timestamp": pd.Timestamp.now().isoformat()
    }

# For backward compatibility
generate_image_description_tool = Image_Analyzer
