"""
ds_engine.time_series.features
==============================

Generate time-based and lag-based features from a datetime column.
This module follows the 3-element return contract:
    run(df, columns, params) -> (output_data, output_plots, enriched_df)

Expected params keys:
    date_column (str): REQUIRED. Name of datetime column.
    lag_periods (list[int]): Default [1, 7, 30].
    rolling_windows (list[int]): Default [7, 30].
    extract_date_parts (bool): Default True.
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
    date_col = params.get('date_column')
    if not date_col:
        raise ValueError("ts_features requires 'date_column' parameter.")
    if date_col not in df.columns:
        raise KeyError(f"date_column '{date_col}' not found in DataFrame.")
        
    lag_periods = params.get('lag_periods', [1, 7, 30])
    rolling_windows = params.get('rolling_windows', [7, 30])
    extract_date_parts = params.get('extract_date_parts', True)
    
    enriched_df = df.copy()
    
    if not pd.api.types.is_datetime64_any_dtype(enriched_df[date_col]):
        enriched_df[date_col] = pd.to_datetime(enriched_df[date_col])
        
    enriched_df = enriched_df.sort_values(date_col).reset_index(drop=True)
    
    target_cols = columns if columns else [c for c in df.columns if c != date_col]
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(enriched_df[c])]
    
    new_cols = []
    
    if extract_date_parts:
        s_dt = enriched_df[date_col].dt
        enriched_df['year'] = s_dt.year
        enriched_df['month'] = s_dt.month
        enriched_df['day'] = s_dt.day
        enriched_df['weekday'] = s_dt.weekday
        enriched_df['quarter'] = s_dt.quarter
        new_cols.extend(['year', 'month', 'day', 'weekday', 'quarter'])
        
    for col in numeric_cols:
        for lag in lag_periods:
            name = f"{col}_lag_{lag}"
            enriched_df[name] = enriched_df[col].shift(lag)
            new_cols.append(name)
            
        for win in rolling_windows:
            name = f"{col}_rolling_mean_{win}"
            enriched_df[name] = enriched_df[col].rolling(window=win, min_periods=1).mean()
            new_cols.append(name)
            
    output_data = {
        'new_columns_added': new_cols,
        'original_shape': [int(df.shape[0]), int(df.shape[1])],
        'new_shape': [int(enriched_df.shape[0]), int(enriched_df.shape[1])]
    }
    
    return output_data, [], enriched_df
