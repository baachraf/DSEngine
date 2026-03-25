"""
ds_engine.data.loader
=====================
Handles data ingestion from CSV, Excel, JSON, and Parquet.
Returns a clean pandas DataFrame.

Expected loader_params keys:
    separator (str): Delimiter. Default: ','.
    encoding (str): Text encoding. Default: 'utf-8'.
    sheet_name (int|str): For Excel. Default: 0.
    orient (str): For JSON. Default: 'records'.
"""

import pandas as pd
from typing import Any
import re
from pathlib import Path

def sanitize_column_name(col: Any) -> str:
    """Sanitize a single column name."""
    col_str = str(col)
    return re.sub(r'[^\w]', '_', col_str.strip().lower()).strip('_')

def sanitize_column_list(cols: list[Any]) -> list[str]:
    """Sanitize a list of column names."""
    if not cols:
        return []
    return [sanitize_column_name(c) for c in cols]

def _sanitize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, replace spaces and special chars with underscores."""
    df = df.copy()
    original_cols = list(df.columns)
    
    new_cols = sanitize_column_list(original_cols)
        
    df.columns = new_cols
    
    # Log column name changes
    try:
        from ds_engine.utils import logger
        log = logger.get_logger('loader')
        for orig, new in zip(original_cols, new_cols):
            if str(orig) != new:
                log.warning(f"Column name changed: '{orig}' -> '{new}'")
    except ImportError:
        pass # In case logger is not available yet
        
    return df

def load(source: str, columns: list[str], params: dict[str, Any]) -> pd.DataFrame:
    """Load data from source file into a pandas DataFrame.
    
    Args:
        source (str): Absolute or relative path to the data file.
        columns (list[str]): List of column names to load. Empty list means all columns.
        params (dict[str, Any]): Loader specific parameters.
            - separator (str): Default ','.
            - encoding (str): Default 'utf-8'.
            - sheet_name (int|str): Default 0.
            - orient (str): Default 'records'.
            
    Returns:
        pd.DataFrame: Sourced and sanitized data.
        
    Raises:
        ValueError: If file format is not supported or encoding fails.
    """
    db_dialects = ('sqlite://', 'mysql://', 'postgresql://', 'oracle://', 'mssql://')
    if source.startswith(db_dialects):
        table = params.get('table')
        query = params.get('query')
        if query:
            try:
                df = pd.read_sql(query, source)
            except Exception as e:
                raise ValueError(f"Failed to execute query on database. Error: {e}")
        elif table:
            try:
                df = pd.read_sql_table(table, source)
            except Exception as e:
                raise ValueError(f"Failed to read table '{table}' from database. Error: {e}")
        else:
            raise ValueError("For database sources, you must provide either 'query' or 'table' in loader_params.")
    else:
        source_path = Path(source)
        if not source_path.exists():
            raise ValueError(f"File not found: {source}")
            
        ext = source_path.suffix.lower()
        
        sep = params.get('separator', ',')
        enc = params.get('encoding', 'utf-8')
        sheet = params.get('sheet_name', 0)
        orient = params.get('orient', 'records')
        
        try:
            if ext == '.csv':
                try:
                    df = pd.read_csv(source_path, sep=sep, encoding=enc)
                except UnicodeDecodeError:
                    # Fallback to latin-1
                    try:
                        from ds_engine.utils import logger
                        log = logger.get_logger('loader')
                        log.warning(f"Failed to decode {source} with {enc}. Falling back to 'latin-1'")
                    except ImportError:
                        pass
                    try:
                        df = pd.read_csv(source_path, sep=sep, encoding='latin-1')
                    except Exception:
                        raise ValueError("File could not be decoded. Try specifying encoding in loader_params (e.g. encoding: cp1252 or encoding: iso-8859-1).")
            elif ext in ['.xlsx', '.xls']:
                df = pd.read_excel(source_path, sheet_name=sheet)
            elif ext == '.json':
                df = pd.read_json(source_path, orient=orient)
            elif ext == '.parquet':
                df = pd.read_parquet(source_path)
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        except Exception as e:
            if isinstance(e, ValueError) and ("Unsupported file format" in str(e) or "File could not be decoded" in str(e)):
                raise
            raise ValueError(f"Failed to read file {source}. Error: {e}")
        
    df = _sanitize_columns(df)
    
    if columns:
        sanitized_columns = sanitize_column_list(columns)
        missing = [c for c in sanitized_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Columns not found in dataset: {columns} (sanitized as {sanitized_columns})")
        df = df[sanitized_columns]
        
    return df
