"""
ds_engine.preparation.encode
============================

Encode categorical features into numeric formats.
This module follows the preparation block contract:
    run(df, columns, params) -> (output_data, output_plots, df_modified)

Expected params keys:
    method (str): 'onehot' or 'label'. Default 'onehot'.
    drop_first (bool): Only for 'onehot'. Default True.
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
    """Encode categorical features.
    
    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Columns to encode. Empty means all object/category.
        params (dict): Step configuration.
        
    Returns:
        tuple[dict, list, pd.DataFrame]: Stats, plots, and the modified DataFrame.
    """
    method = params.get('method', 'onehot')
    df_new = df.copy()
    
    target_cols = columns if columns else [c for c in df_new.columns if pd.api.types.is_object_dtype(df_new[c]) or hasattr(df_new[c], 'cat')]
    
    output_data = {
        'method': method,
        'encoded_columns': [],
        'new_columns_added': []
    }
    
    if method == 'onehot':
        drop_first = params.get('drop_first', True)
        original_cols = set(df_new.columns)
        df_new = pd.get_dummies(df_new, columns=target_cols, drop_first=drop_first)
        # pd.get_dummies might create column names with spaces or special chars depending on category names.
        # We should sanitize the new columns.
        new_cols = [c for c in df_new.columns if c not in original_cols or c in target_cols] # target_cols are removed by get_dummies
        
        # Actually pd.get_dummies removes the original columns.
        # Let's clean all new columns.
        current_cols = set(df_new.columns)
        new_ones = list(current_cols - (original_cols - set(target_cols)))
        
        # Sanitize new column names to follow library standard
        mapping = {c: '_'.join(str(c).split()).lower() for c in new_ones} # Basic sanitization
        df_new = df_new.rename(columns=mapping)
        
        output_data['encoded_columns'] = target_cols
        output_data['new_columns_added'] = list(mapping.values())
        
    elif method == 'label':
        for col in target_cols:
            df_new[col] = df_new[col].astype('category').cat.codes
            output_data['encoded_columns'].append(col)
    else:
        raise ValueError(f"Unknown encoding method: {method}")
        
    return output_data, [], df_new
