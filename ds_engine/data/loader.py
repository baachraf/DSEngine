"""
ds_engine.data.loader
=====================

Accept a file path in any supported format and return a clean pandas DataFrame.
This module does not follow the block contract. Called directly by pipeline_runner.
Expected params keys:
    separator (str): Default ','.
    encoding (str): Default 'utf-8'.
    sheet_name (int/str): Default 0.
    orient (str): Default 'records'.
"""

import sys
import pandas as pd
import re
from pathlib import Path
from typing import Any
from ds_engine.utils import logger

def sanitize_column_name(col: Any) -> str:
    """Lowercase + replace non-word chars, strip leading/trailing underscores."""
    return re.sub(r'[^\w]', '_', str(col).strip().lower()).strip('_')

def sanitize_column_list(columns: list[str]) -> list[str]:
    """Apply sanitize_column_name to a list of strings."""
    if not columns:
        return []
    return [sanitize_column_name(c) for c in columns]

def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, replace spaces and special chars with underscores.
    Logs any name changes so the user knows what to put in pipeline.yml.
    """
    df = df.copy()
    new_cols = []
    for col in df.columns:
        original = col
        sanitized = sanitize_column_name(col)
        
        if sanitized != original:
            logger.log_warning(f"Column name changed: '{original}' -> '{sanitized}'")
        new_cols.append(sanitized)
        
    df.columns = new_cols
    return df

def load(
    source: str, 
    columns: list[str], 
    params: dict[str, Any]
) -> pd.DataFrame:
    """Load a dataset into a pandas DataFrame.
    
    Args:
        source (str): Absolute file path to the dataset.
        columns (list[str]): List of columns to load. If empty, load all.
        params (dict): Load configuration (separator, encoding, etc).
        
    Returns:
        pd.DataFrame: The loaded DataFrame.
        
    Raises:
        SystemExit: On FATAL errors (file missing, format not supported, decoding err).
    """
    path = Path(source)
    if not path.is_file():
        logger.log_error(f"[FATAL] Data source file not found.\n"
                         f"Caused by: resolved path '{source}' does not exist.")
        sys.exit(2)
        
    ext = path.suffix.lower()
    sep = params.get('separator', ',')
    enc = params.get('encoding', 'utf-8')
    sheet = params.get('sheet_name', 0)
    orient = params.get('orient', 'records')
    
    try:
        if ext == '.csv':
            try:
                df = pd.read_csv(source, sep=sep, encoding=enc)
            except UnicodeDecodeError:
                logger.log_warning(f"File failed to decode with '{enc}'. Falling back to 'latin-1'.")
                try:
                    df = pd.read_csv(source, sep=sep, encoding='latin-1')
                except Exception:
                    logger.log_error(
                        f"[FATAL] File could not be decoded. "
                        f"Try specifying encoding in loader_params (e.g. encoding: cp1252 or encoding: iso-8859-1)."
                    )
                    sys.exit(2)
        elif ext in ['.xlsx', '.xls']:
            df = pd.read_excel(source, sheet_name=sheet)
        elif ext == '.json':
            df = pd.read_json(source, orient=orient)
        elif ext == '.parquet':
            df = pd.read_parquet(source)
        else:
            logger.log_error(f"[FATAL] File format not supported for '{path.name}'.")
            sys.exit(2)
            
    except Exception as e:
        logger.log_error(f"[FATAL] File exists but cannot be read: {e}")
        sys.exit(2)
        
    # Apply column sanitization
    df = _sanitize_columns(df)
    
    # Filter columns if requested
    if columns:
        # Sanitize requested columns to match normalized DataFrame columns
        sanitized_req = sanitize_column_list(columns)
        missing = [c for idx, c in enumerate(sanitized_req) if c not in df.columns]
        if missing:
            # Show original names in error message for better UX
            orig_missing = [columns[idx] for idx, c in enumerate(sanitized_req) if c not in df.columns]
            logger.log_error(f"[FATAL] Requested columns not found in dataset: {orig_missing}")
            sys.exit(2)
        df = df[sanitized_req]
        
    return df
