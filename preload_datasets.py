# preload_datasets.py
import scanpy as sc
import pandas as pd
import os
from dotenv import load_dotenv
from utils import parse_tsv_data
import asyncpg
from psycopg_pool import AsyncConnectionPool

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
DATABASE_URL = os.getenv("DATABASE_URL")
TRAIN_DATA_FILE = os.getenv("TRAIN_DATA_FILE")
TRAIN_IMAGE_DATA_FILE = os.getenv("TRAIN_IMAGE_DATA_FILE")
try:
    BASE_URL = os.environ["BASE_URL"]
except KeyError:
    raise ValueError("BASE_URL environment variable is not set. Please set it in your .env file.")

# PostgreSQL connection configuration
connection_kwargs = {
    "autocommit": True,
    "prepare_threshold": 0,
}

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

# PostgreSQL Connection Pool
async def get_db_pool():
    try:
        pool = await AsyncConnectionPool(
            conninfo=DATABASE_URL,
            max_size=20,
            kwargs=connection_kwargs,
        ).__aenter__()
        return pool
    except Exception as e:
        print(f"Error creating PostgreSQL connection pool: {e}")
        return None

# Context manager for database connection
class AsyncDBConnection:
    def __init__(self):
        self.pool = None
        self.conn = None

    async def __aenter__(self):
        if not self.pool:
            self.pool = await get_db_pool()
        if self.pool:
            self.conn = await self.pool.acquire()
            return self.conn
        return None

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.conn and self.pool:
            await self.pool.release(self.conn)
            self.conn = None

# Use this instead of the global conn variable
db = AsyncDBConnection()
