"""
ds_engine.preparation.scale
===========================

Scale numeric features in the DataFrame.
This module follows the preparation block contract:
    run(df, columns, params) -> (output_data, output_plots, df_modified)

Expected params keys:
    method (str): 'standard', 'minmax', or 'robust'. Default 'standard'.
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
    """Scale numeric features.
    
    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Numeric columns to scale. Empty means all numeric.
        params (dict): Step configuration.
        
    Returns:
        tuple[dict, list, pd.DataFrame]: Stats, plots, and the modified DataFrame.
    """
    method = params.get('method', 'standard')
    df_new = df.copy()
    
    target_cols = columns if columns else [c for c in df_new.columns if pd.api.types.is_numeric_dtype(df_new[c])]
    
    output_data = {
        'method': method,
        'scaled_columns': []
    }
    
    for col in target_cols:
        s = df_new[col].dropna()
        if len(s) == 0:
            continue
            
        if method == 'standard':
            mu, std = s.mean(), s.std()
            if std > 0:
                df_new[col] = (df_new[col] - mu) / std
        elif method == 'minmax':
            cmin, cmax = s.min(), s.max()
            if cmax > cmin:
                df_new[col] = (df_new[col] - cmin) / (cmax - cmin)
        elif method == 'robust':
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            med = s.median()
            iqr = q3 - q1
            if iqr > 0:
                df_new[col] = (df_new[col] - med) / iqr
        else:
            raise ValueError(f"Unknown scaling method: {method}")
            
        output_data['scaled_columns'].append(col)
        
    return output_data, [], df_new
