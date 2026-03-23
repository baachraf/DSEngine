"""
ds_engine.exploration.summary
=============================

Generate a complete statistical summary of numeric columns.
This module follows the standard block contract:
    run(df, columns, params) -> (output_data, output_plots)

Expected params keys:
    percentiles (list[float]): Default [0.25, 0.5, 0.75].
    include_skew (bool): Default True.
    include_kurtosis (bool): Default True.
"""

import pandas as pd
import matplotlib.figure
from typing import Any
from ds_engine.utils import logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Generate summary statistics for numeric columns.
    
    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Columns to analyze. Empty means all numeric.
        params (dict): Step configuration.
        
    Returns:
        tuple[dict, list]: Output data dict and empty plots list.
    """
    percentiles = params.get('percentiles', [0.25, 0.5, 0.75])
    include_skew = params.get('include_skew', True)
    include_kurtosis = params.get('include_kurtosis', True)
    
    target_cols = columns if columns else df.columns
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    # Log omitted columns
    skipped = set(target_cols) - set(numeric_cols)
    if skipped:
        logger.log_warning(f"summary skipped non-numeric columns: {list(skipped)}")
        
    output_data: dict[str, Any] = {}
    
    for col in numeric_cols:
        s = df[col]
        valid_s = s.dropna()
        n_total = len(s)
        n_valid = len(valid_s)
        n_null = int(s.isna().sum())
        
        col_stats: dict[str, Any] = {
            'count': int(n_valid),
            'null_count': n_null,
            'null_pct': float(n_null / n_total) * 100 if n_total > 0 else 0.0,
            'dtype': str(s.dtype),
            'unique_count': int(valid_s.nunique())
        }
        
        if n_valid > 0:
            col_stats['mean'] = float(valid_s.mean())
            col_stats['std'] = float(valid_s.std())
            col_stats['min'] = float(valid_s.min())
            col_stats['max'] = float(valid_s.max())
            
            # Percentiles
            col_stats['percentiles'] = {}
            for pct in percentiles:
                col_stats['percentiles'][f"{pct*100:g}%"] = float(valid_s.quantile(pct))
                
            if include_skew and n_valid > 2:
                col_stats['skewness'] = float(valid_s.skew())
            if include_kurtosis and n_valid > 3:
                col_stats['kurtosis'] = float(valid_s.kurtosis())
                
        output_data[col] = col_stats
        
    return output_data, []
