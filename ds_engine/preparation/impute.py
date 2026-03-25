"""
ds_engine.preparation.impute
============================

Impute missing values in the DataFrame.
This module follows the preparation block contract:
    run(df, columns, params) -> (output_data, output_plots, df_modified)

Expected params keys:
    method (str): 'mean', 'median', 'mode', or 'constant'. Default 'mean'.
    fill_value (any): Used if method is 'constant'. Default None.
"""

import pandas as pd
import matplotlib.figure
from typing import Any
from ds_engine.utils import logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure], pd.DataFrame]:
    """Impute missing values in specified columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Columns to impute. Empty means all with nulls.
        params (dict): Step configuration.
        
    Returns:
        tuple[dict, list, pd.DataFrame]: Stats, plots, and the modified DataFrame.
    """
    method = params.get('method', 'mean')
    constant_val = params.get('fill_value')
    
    df_new = df.copy()
    target_cols = columns if columns else df.columns.tolist()
    
    output_data = {
        'method': method,
        'imputed_columns': {}
    }
    
    for col in target_cols:
        null_count = int(df_new[col].isna().sum())
        if null_count == 0:
            continue
            
        if method == 'mean':
            if pd.api.types.is_numeric_dtype(df_new[col]):
                fill_val = df_new[col].mean()
            else:
                logger.log_warning(f"Cannot use 'mean' imputation on non-numeric column '{col}'. Skipping.")
                continue
        elif method == 'median':
            if pd.api.types.is_numeric_dtype(df_new[col]):
                fill_val = df_new[col].median()
            else:
                logger.log_warning(f"Cannot use 'median' imputation on non-numeric column '{col}'. Skipping.")
                continue
        elif method == 'mode':
            mode_res = df_new[col].mode()
            fill_val = mode_res[0] if not mode_res.empty else None
        elif method == 'constant':
            fill_val = constant_val
        else:
            raise ValueError(f"Unknown imputation method: {method}")
            
        if fill_val is not None:
            df_new[col] = df_new[col].fillna(fill_val)
            output_data['imputed_columns'][col] = {
                'null_count_fixed': null_count,
                'fill_value': float(fill_val) if isinstance(fill_val, (int, float)) else str(fill_val)
            }
            
    return output_data, [], df_new
