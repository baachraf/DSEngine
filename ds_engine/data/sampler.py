"""
ds_engine.data.sampler
======================

Returns a representative subset of the DataFrame.
This module follows a modified block contract handling a third return element:
    run(df, columns, params) -> (output_data, output_plots, sampled_df)

Expected params keys:
    method (str): 'random' or 'stratified'. Default: 'random'.
    size (float/int): Fraction (0-1) or absolute number. Default: 0.1.
    stratify_column (str): Required if method is 'stratified'.
    random_state (int): Default: 42.
"""

import pandas as pd
import matplotlib.figure
from typing import Any

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure], pd.DataFrame]:
    """Sample the input DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame. Never modified in-place.
        columns (list[str]): Ignored for sampling (samples rows across all columns).
        params (dict): Step-specific parameters.
        
    Returns:
        tuple[dict, list, pd.DataFrame]: (output_data, output_plots, sampled_df)
    """
    method = params.get('method', 'random')
    size = params.get('size', 0.1)
    random_state = params.get('random_state', 42)
    stratify_col = params.get('stratify_column')

    n_rows = len(df)
    is_frac = isinstance(size, float) and 0.0 < size < 1.0
    
    # If absolute size is larger than dataset, cap it
    if not is_frac and size >= n_rows:
        sampled_df = df.copy()
    else:
        if method == 'stratified':
            if not stratify_col:
                raise ValueError("sampler requires 'stratify_column' when method='stratified'")
            if stratify_col not in df.columns:
                raise KeyError(f"stratify_column '{stratify_col}' not found in DataFrame")
                
            if is_frac:
                sampled_df = df.groupby(stratify_col, group_keys=False).apply(
                    lambda x: x.sample(frac=size, random_state=random_state)
                ).reset_index(drop=True)
            else:
                # Approximate stratified exact amount
                frac = float(size) / n_rows
                sampled_df = df.groupby(stratify_col, group_keys=False).apply(
                    lambda x: x.sample(frac=frac, random_state=random_state)
                ).reset_index(drop=True)
        else:
            if is_frac:
                sampled_df = df.sample(frac=size, random_state=random_state).reset_index(drop=True)
            else:
                sampled_df = df.sample(n=int(size), random_state=random_state).reset_index(drop=True)

    output_data = {
        'original_shape': [int(df.shape[0]), int(df.shape[1])],
        'sampled_shape': [int(sampled_df.shape[0]), int(sampled_df.shape[1])],
        'method_used': method,
        'stratify_column': stratify_col
    }
    
    return output_data, [], sampled_df
