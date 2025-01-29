import scanpy as sc
import pandas as pd
import sqlite3
import os
from dotenv import load_dotenv
from utils import parse_tsv_data

# Load environment variables from the .env file
load_dotenv()

# Global dictionary for preloaded data
PRELOADED_DATA = {}

# Preloaded dataset index
PRELOADED_DATASET_INDEX = None

# Retrieve paths from environment variables
BASE_DATASET_DIR = os.getenv("BASE_DATASET_DIR")
PLOT_OUTPUT_DIR = os.getenv("PLOT_OUTPUT_DIR")
DATASET_INDEX_FILE = os.getenv("DATASET_INDEX_FILE")
DATABASE_PATH = os.getenv("DATABASE_PATH")
TRAIN_DATA_FILE = os.getenv("TRAIN_DATA_FILE")

def preload_dataset_index(file_path: str):
    global PRELOADED_DATASET_INDEX
    try:
        # Use the parse_tsv_data function to get a JSON-serializable dict
        PRELOADED_DATASET_INDEX = parse_tsv_data(file_path)
        print("Dataset index successfully preloaded into memory as a serializable object.")
    except Exception as e:
        print(f"Error preloading dataset index: {e}")

def preload_all_h5ad_files(base_dir: str):
    global PRELOADED_DATA
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory {base_dir} does not exist.")
        return
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue
        for file in os.listdir(subdir_path):
            if file.endswith(".h5ad"):
                fullpath = os.path.join(subdir_path, file)
                if fullpath not in PRELOADED_DATA:
                    print(f"Preloading {fullpath}...")
                    try:
                        PRELOADED_DATA[fullpath] = sc.read_h5ad(fullpath)
                        print(f"Successfully preloaded {file}.")
                    except Exception as e:
                        print(f"Error loading {file}: {e}")

# Preload the dataset index
if DATASET_INDEX_FILE:
    preload_dataset_index(DATASET_INDEX_FILE)
else:
    print("Error: DATASET_INDEX_FILE is not set in the environment.")

# Preload all .h5ad files
if BASE_DATASET_DIR:
    preload_all_h5ad_files(BASE_DATASET_DIR)
else:
    print("Error: BASE_DATASET_DIR is not set in the environment.")

# Connect to the SQLite database
if DATABASE_PATH:
    try:
        conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        print("Successfully connected to the SQLite database.")
    except Exception as e:
        print(f"Error connecting to the database: {e}")
else:
    print("Error: DATABASE_PATH is not set in the environment.")
