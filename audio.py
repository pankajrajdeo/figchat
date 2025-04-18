import os
import io
import asyncio
import numpy as np
import audioop
import soundfile as sf
from uuid import uuid4
import chainlit as cl
from dotenv import load_dotenv
import json
from datetime import datetime
from groq import Groq

# Load environment variables
load_dotenv()

# Initialize Groq client (optional, as it's defined in app.py, but kept for completeness)
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Path to the log file for Whisper model usage
LOG_FILE_PATH = "whisper_usage_log.json"

# Whisper models configuration
WHISPER_MODELS_CONFIG = [
    {"name": "whisper-large-v3-turbo", "daily_limit": 2000},
    {"name": "distil-whisper-large-v3-en", "daily_limit": 2000},
    {"name": "whisper-large-v3", "daily_limit": 2000},
]

# Function to load usage data from log file
def load_usage_data():
    if os.path.exists(LOG_FILE_PATH):
        try:
            with open(LOG_FILE_PATH, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error loading usage log: {e}. Creating new log.")
    
    # Initialize with default data if file doesn't exist or is corrupted
    return {model["name"]: {"requests_today": 0, "last_reset": datetime.now().isoformat()} 
            for model in WHISPER_MODELS_CONFIG}

# Function to save usage data to log file
def save_usage_data(usage_data):
    try:
        with open(LOG_FILE_PATH, 'w') as f:
            json.dump(usage_data, f, indent=2)
    except Exception as e:
        print(f"Error saving usage log: {e}")

# Function to get the current available Whisper model
def get_current_whisper_model():
    usage_data = load_usage_data()
    now = datetime.now()
    
    for model_name, data in usage_data.items():
        last_reset = datetime.fromisoformat(data["last_reset"])
        if (now - last_reset).days > 0:
            usage_data[model_name]["requests_today"] = 0
            usage_data[model_name]["last_reset"] = now.isoformat()
    
    save_usage_data(usage_data)
    
    for model in WHISPER_MODELS_CONFIG:
        model_name = model["name"]
        if usage_data[model_name]["requests_today"] < model["daily_limit"]:
            return model_name
    
    print("WARNING: All Whisper models have reached their daily limits!")
    return WHISPER_MODELS_CONFIG[0]["name"]

# Function to increment the request count for a model
def increment_model_requests(model_name):
    usage_data = load_usage_data()
    
    if model_name in usage_data:
        usage_data[model_name]["requests_today"] += 1
        requests = usage_data[model_name]["requests_today"]
        
        daily_limit = next((model["daily_limit"] for model in WHISPER_MODELS_CONFIG if model["name"] == model_name), 1900)
        
        print(f"Model {model_name} has used {requests}/{daily_limit} requests today")
        
        save_usage_data(usage_data)

# Function to display current model usage statistics
def display_model_usage_stats():
    usage_data = load_usage_data()
    print("\n=== WHISPER MODEL USAGE STATISTICS ===")
    print(f"Log file: {LOG_FILE_PATH}")
    print("-" * 40)
    
    for model in WHISPER_MODELS_CONFIG:
        model_name = model["name"]
        if model_name in usage_data:
            data = usage_data[model_name]
            last_reset = datetime.fromisoformat(data["last_reset"])
            requests = data["requests_today"]
            limit = model["daily_limit"]
            remaining = limit - requests
            percent = (requests / limit) * 100 if limit > 0 else 0
            
            print(f"Model: {model_name}")
            print(f"  Requests today: {requests}/{limit} ({percent:.1f}%)")
            print(f"  Remaining: {remaining}")
            print(f"  Last reset: {last_reset.strftime('%Y-%m-%d %H:%M:%S')}")
            print("-" * 40)
    
    print("")

# Speech-to-Text function
async def speech_to_text(audio_file):
    try:
        loop = asyncio.get_event_loop()
        
        def transcribe():
            current_model = get_current_whisper_model()
            print(f"Using Whisper model: {current_model}")
            
            response = groq_client.audio.transcriptions.create(file=audio_file, model=current_model)
            
            increment_model_requests(current_model)
            
            return response
            
        response = await loop.run_in_executor(None, transcribe)
        return response.text
    except Exception as e:
        print(f"Error in speech_to_text: {e}")
        return "I couldn't transcribe the audio. Please try again."

# Process audio function
async def process_audio():
    try:
        audio_chunks = cl.user_session.get("audio_chunks", [])
        if not audio_chunks:
            print("Warning: No audio chunks to process")
            return None
            
        combined_audio = np.concatenate(audio_chunks)
        if len(combined_audio) == 0:
            print("Warning: Combined audio is empty")
            return None
            
        wav_buffer = io.BytesIO()
        with sf.SoundFile(wav_buffer, 'w', samplerate=24000, channels=1, format='WAV') as wav_file:
            wav_file.write(combined_audio)
        wav_buffer.seek(0)
        cl.user_session.set("audio_chunks", [])

        whisper_input = ("audio.wav", wav_buffer.getvalue(), "audio/wav")
        transcription = await speech_to_text(whisper_input)
        
        if not transcription or transcription.strip() == "":
            print("Warning: Empty transcription result")
            await cl.Message(author="System", content="I couldn't hear anything. Please try speaking again.").send()
            return None
            
        return transcription  # Remove the message sending
    except Exception as e:
        print(f"Error in process_audio: {e}")
        await cl.Message(author="System", content="An error occurred while processing your audio. Please try again.").send()
        return None
