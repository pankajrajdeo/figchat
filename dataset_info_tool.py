import os
import json
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from typing import Dict, Optional, Union

# Load environment variables
env_loaded = load_dotenv()
if not env_loaded:
    raise ValueError(".env file not found or failed to load.")

try:
    BASE_URL = os.environ["BASE_URL"]
except KeyError:
    raise ValueError("BASE_URL environment variable is not set. Please set it in your .env file.")

BASE_DATASET_DIR = os.getenv('BASE_DATASET_DIR')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PLOT_OUTPUT_DIR = os.getenv('PLOT_OUTPUT_DIR')

if not BASE_DATASET_DIR:
    raise ValueError("BASE_DATASET_DIR environment variable is not set. Please set it in your .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set. Please set it in your .env file.")

# Constants for context management
MAX_SIZE_FOR_FULL_LOAD_MB = 1  # MB threshold for large files
MAX_ROWS_FOR_LLM_CONTEXT = 500
MAX_COLS_FOR_LLM_CONTEXT = 10


def resolve_tsv_path(tsv_path: str) -> str:
    """
    Resolve the TSV file path, handling both full URLs and paths relative to BASE_DATASET_DIR.
    """
    if tsv_path.startswith(BASE_URL):
        relative = "/" + tsv_path[len(BASE_URL):].lstrip("/")
    else:
        relative = tsv_path
    if os.path.isabs(relative):
        return relative
    if PLOT_OUTPUT_DIR:
        candidate = os.path.join(PLOT_OUTPUT_DIR, os.path.basename(relative))
        if os.path.exists(candidate):
            return candidate
    return os.path.join(BASE_DATASET_DIR, relative)


def is_large_file(path: str) -> (bool, float):
    try:
        size_bytes = os.path.getsize(path)
        size_mb = size_bytes / (1024 * 1024)
        return size_mb >= MAX_SIZE_FOR_FULL_LOAD_MB, size_mb
    except OSError:
        return False, 0.0


def summarize_large_file(path: str) -> Dict:
    """
    Summarize a large TSV by capturing its dimensions, columns, and first few rows.
    """
    size_bytes = os.path.getsize(path)
    size_mb = size_bytes / (1024 * 1024)
    cols = list(pd.read_csv(path, sep='\t', nrows=0).columns)
    head5 = pd.read_csv(path, sep='\t', nrows=5).to_dict(orient='records')
    # estimate row count by line count
    try:
        with open(path, 'r') as f:
            total_rows = sum(1 for _ in f) - 1
    except Exception:
        total_rows = None
    return {
        "file_name": os.path.basename(path),
        "file_size_mb": round(size_mb, 2),
        "columns": cols,
        "column_count": len(cols),
        "estimated_row_count": total_rows,
        "first_rows": head5
    }


def apply_transformations(df: pd.DataFrame, instr: Dict) -> pd.DataFrame:
    """
    Apply filtering, range, and sorting based on instructions.
    Supported types: head, tail, column_value, column_contains, range, sort
    """
    t = instr.get('type')
    if t == 'head':
        n = instr.get('rows', MAX_ROWS_FOR_LLM_CONTEXT)
        return df.head(n)
    if t == 'tail':
        n = instr.get('rows', MAX_ROWS_FOR_LLM_CONTEXT)
        return df.tail(n)
    if t == 'column_value':
        col, val = instr.get('column'), instr.get('value')
        return df[df.get(col) == val] if col in df.columns else df.iloc[0:0]
    if t == 'column_contains':
        col, val = instr.get('column'), instr.get('value')
        return df[df[col].astype(str).str.contains(val, na=False)] if col in df.columns else df.iloc[0:0]
    if t == 'range':
        col = instr.get('column')
        mn, mx = instr.get('min'), instr.get('max')
        if col not in df.columns:
            return df.iloc[0:0]
        if mn is not None and mx is not None:
            return df[(df[col] >= mn) & (df[col] <= mx)]
        if mn is not None:
            return df[df[col] >= mn]
        if mx is not None:
            return df[df[col] <= mx]
        return df
    if t == 'sort':
        col = instr.get('column')
        asc = instr.get('ascending', True)
        limit = instr.get('limit')
        if col in df.columns:
            sorted_df = df.sort_values(col, ascending=asc)
            if isinstance(limit, int):
                return sorted_df.head(limit)
            return sorted_df
        return df
    return df

async def analyze_with_llm(data: Union[Dict, pd.DataFrame], query: str, context_msg: str = "") -> str:
    """
    Send data and query to the LLM for analysis.
    """
    if isinstance(data, pd.DataFrame):
        data_str = data.to_markdown(index=False) if not data.empty else "Dataset is empty."
    else:
        data_str = json.dumps(data, indent=2)
    prompt = (
        f"Analyze the following data and answer the query concisely.\n"
        f"Context: {context_msg}\nQuery: {query}\nData:\n{data_str}\n"
    )
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        google_api_key=GOOGLE_API_KEY,
        temperature=0
    )
    msg = HumanMessage(content=prompt)
    resp = model.invoke([msg])
    return resp.content.strip()

async def process_and_analyze_tsv(resolved_path: str, query: str, filter_instructions: Optional[Dict] = None) -> str:
    if not os.path.exists(resolved_path):
        return json.dumps({"error": f"File not found: {resolved_path}"})
    large, size_mb = is_large_file(resolved_path)
    # Large file handling
    if large:
        summary = summarize_large_file(resolved_path)
        if filter_instructions:
            # apply on entire file
            df_full = pd.read_csv(resolved_path, sep='\t')
            df_filtered = apply_transformations(df_full, filter_instructions)
            r, c = df_filtered.shape
            if r <= MAX_ROWS_FOR_LLM_CONTEXT and c <= MAX_COLS_FOR_LLM_CONTEXT:
                ctx = (f"Showing filtered data ({r} rows x {c} cols) from large file (~{summary['estimated_row_count']} rows, "
                       f"{size_mb:.2f} MB): {os.path.basename(resolved_path)}")
                analysis = await analyze_with_llm(df_filtered, query, ctx)
                result = {
                    "tsv_path": resolved_path,
                    "file_size_mb": round(size_mb,2),
                    "query": query,
                    "filter_applied": filter_instructions,
                    "context_message": ctx,
                    "analysis": analysis
                }
                return json.dumps(result, indent=4)
            else:
                # write out filtered file
                base, ext = os.path.splitext(resolved_path)
                out_path = f"{base}_filtered{ext}"
                df_filtered.to_csv(out_path, sep='\t', index=False)
                result = {
                    "message": "Filtered data too large for LLM context. Download the filtered file below.",
                    "filtered_file": out_path,
                    "filtered_rows": r,
                    "filtered_cols": c,
                    "original_file_summary": summary
                }
                return json.dumps(result, indent=4)
        # No filter: just return summary
        return json.dumps({"summary": summary}, indent=4)
    # Small file handling
    df = pd.read_csv(resolved_path, sep='\t')
    total_rows, total_cols = df.shape
    if filter_instructions:
        df_filtered = apply_transformations(df, filter_instructions)
        r, c = df_filtered.shape
        if r > MAX_ROWS_FOR_LLM_CONTEXT or c > MAX_COLS_FOR_LLM_CONTEXT:
            df_context = df_filtered.head(MAX_ROWS_FOR_LLM_CONTEXT).iloc[:, :MAX_COLS_FOR_LLM_CONTEXT]
            ctx = (f"Showing first {MAX_ROWS_FOR_LLM_CONTEXT} rows and {MAX_COLS_FOR_LLM_CONTEXT} cols of filtered data "
                   f"({r} matching rows) from file ({total_rows} rows x {total_cols} cols): {os.path.basename(resolved_path)}")
            data_for_llm = df_context
        else:
            ctx = (f"Showing filtered data ({r} rows x {c} cols) from file ({total_rows} rows x {total_cols} cols): "
                   f"{os.path.basename(resolved_path)}")
            data_for_llm = df_filtered
    else:
        if total_rows > MAX_ROWS_FOR_LLM_CONTEXT or total_cols > MAX_COLS_FOR_LLM_CONTEXT:
            data_for_llm = df.head(MAX_ROWS_FOR_LLM_CONTEXT).iloc[:, :MAX_COLS_FOR_LLM_CONTEXT]
            ctx = (f"Showing head {MAX_ROWS_FOR_LLM_CONTEXT} rows and first {MAX_COLS_FOR_LLM_CONTEXT} cols of file "
                   f"({total_rows} rows x {total_cols} cols): {os.path.basename(resolved_path)}")
        else:
            data_for_llm = df
            ctx = (f"Showing all data ({total_rows} rows x {total_cols} cols) from file: "
                   f"{os.path.basename(resolved_path)}")
    analysis = await analyze_with_llm(data_for_llm, query, ctx)
    result = {
        "tsv_path": resolved_path,
        "file_size_mb": round(size_mb,2),
        "query": query,
        "filter_applied": filter_instructions,
        "context_message": ctx,
        "analysis": analysis
    }
    return json.dumps(result, indent=4)

async def Dataset_Explorer(query: str = "", tsv_path: str = None, filter_instructions: Optional[Dict] = None) -> str:
    """
    Provides dataset information and analyzes TSV files based on the context of the query.
    """
    if tsv_path:
        if not query:
            return "Error: Please provide a query when analyzing a TSV file."
        resolved = resolve_tsv_path(tsv_path)
        return await process_and_analyze_tsv(resolved, query, filter_instructions)
    # Fallback metadata route
    try:
        from visualization_tool import PRELOADED_DATASET_INDEX
        if PRELOADED_DATASET_INDEX is None:
            return "Error: Dataset index not preloaded."
        data = PRELOADED_DATASET_INDEX
        datasets = data.get("datasets", [])
        notes = data.get("notes", {})
        for ds in datasets:
            ds.pop("**Directory Path** (File location of the dataset)", None)
        return json.dumps({"datasets": datasets, "notes": notes}, indent=4)
    except Exception as e:
        return json.dumps({"error": str(e)})

dataset_info_tool = Dataset_Explorer
