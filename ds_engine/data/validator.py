"""
ds_engine.data.validator
========================

Validates a DataFrame against a user-defined schema.
This module follows the standard block contract:
    run(df, columns, params) -> (output_data, output_plots)
    
Expected params keys:
    schema_path (str): Absolute path to the YAML schema file.
"""

import yaml
import pandas as pd
import matplotlib.figure
from typing import Any
from ds_engine.utils import logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Validate DataFrame against a schema.
    
    Args:
        df (pd.DataFrame): The input DataFrame. Never modified in-place.
        columns (list[str]): List of column names to validate. If empty, uses df columns.
        params (dict): Config parameters. Must contain 'schema_path'.
        
    Returns:
        tuple[dict, list]: A tuple of (output_data, output_plots).
        
    Raises:
        ValueError: If 'schema_path' is missing or points to a non-existent file.
    """
    schema_path = params.get('schema_path')
    if not schema_path:
        raise ValueError("Missing 'schema_path' parameter in validate step. "
                         "Please specify a path to schema_template.yml.")
                         
    try:
        with open(schema_path, 'r', encoding='utf-8') as f:
            schema_config = yaml.safe_load(f)
    except FileNotFoundError:
        raise ValueError(f"Schema file not found at: '{schema_path}'")
        
    schema_columns = schema_config.get('columns', {})
    
    validate_cols = columns if columns else list(df.columns)
    
    output_data: dict[str, Any] = {
        'is_valid': True,
        'column_results': {}
    }
    
    for col in validate_cols:
        col_res: dict[str, Any] = {
            'passed': True,
            'dtype_match': True,
            'null_count': int(df[col].isna().sum()),
            'constraint_violations': []
        }
        
        if col not in schema_columns:
            output_data['column_results'][col] = col_res
            continue
            
        rules = schema_columns[col]
        
        # Check dtype
        expected_dtype = rules.get('dtype')
        if expected_dtype:
            actual_dtype = str(df[col].dtype)
            # Relaxed matching for generalized dtypes (e.g., matching 'float64' to 'float64', or 'object' to 'O')
            if expected_dtype not in actual_dtype and actual_dtype not in expected_dtype:
                if not (expected_dtype == 'object' and actual_dtype == 'O'):
                    col_res['dtype_match'] = False
                    col_res['passed'] = False
                    col_res['constraint_violations'].append(f"dtype mismatch: expected {expected_dtype}, got {actual_dtype}")
        
        # Check nullable
        nullable = rules.get('nullable', True)
        if not nullable and col_res['null_count'] > 0:
            col_res['passed'] = False
            col_res['constraint_violations'].append(f"Contains {col_res['null_count']} null values, but nullable=False")
            
        valid_s = df[col].dropna()
        
        if len(valid_s) > 0:
            # Check min
            col_min = rules.get('min')
            if col_min is not None and pd.api.types.is_numeric_dtype(valid_s):
                min_val = valid_s.min()
                if min_val < col_min:
                    col_res['passed'] = False
                    col_res['constraint_violations'].append(f"Minimum value {min_val} < allowed min {col_min}")
            
            # Check max
            col_max = rules.get('max')
            if col_max is not None and pd.api.types.is_numeric_dtype(valid_s):
                max_val = valid_s.max()
                if max_val > col_max:
                    col_res['passed'] = False
                    col_res['constraint_violations'].append(f"Maximum value {max_val} > allowed max {col_max}")
            
            # Check allowed values
            allowed_vals = rules.get('allowed_values')
            if allowed_vals is not None:
                invalid_vals = valid_s[~valid_s.isin(allowed_vals)]
                if len(invalid_vals) > 0:
                    col_res['passed'] = False
                    col_res['constraint_violations'].append(f"Contains {len(invalid_vals)} values outside allowed_values list")

        if not col_res['passed']:
            output_data['is_valid'] = False
            
        output_data['column_results'][col] = col_res
        
    return output_data, []  # Validator produces no plots
