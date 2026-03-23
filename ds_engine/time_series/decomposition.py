"""
ds_engine.time_series.decomposition
===================================

Decompose a time series column into trend, seasonality, and residual components.
This module follows the standard block contract.

Expected params keys:
    date_column (str): REQUIRED. Name of the datetime column.
    model (str): 'additive' or 'multiplicative'. Default 'additive'.
    period (int): Optional. Auto-detected if null.
"""

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.figure
from typing import Any
from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    date_col = params.get('date_column')
    if not date_col:
        raise ValueError("decomposition requires 'date_column' parameter.")
    if date_col not in df.columns:
        raise KeyError(f"date_column '{date_col}' not found in DataFrame.")
        
    model = params.get('model', 'additive')
    period = params.get('period')
    
    # Sort and set index without modifying input df inplace
    ts_df = df.copy()
    
    # Ensure datetime format
    if not pd.api.types.is_datetime64_any_dtype(ts_df[date_col]):
        try:
            ts_df[date_col] = pd.to_datetime(ts_df[date_col])
        except Exception:
            raise TypeError(f"Could not convert '{date_col}' to datetime.")
            
    ts_df = ts_df.sort_values(date_col).set_index(date_col)
    
    if period is None:
        freq = pd.infer_freq(ts_df.index)
        if freq is None:
            raise ValueError(
                f"Could not infer time frequency for '{date_col}'. "
                "Please set the 'period' parameter explicitly in pipeline.yml."
            )
            
        period_map = {'D': 7, 'W': 52, 'M': 12, 'MS': 12, 'Q': 4, 'QS': 4, 'H': 24, 'B': 5}
        # Check against base frequencies ignoring multipliers (e.g. '2H')
        base_freq = ''.join([c for c in freq if c.isalpha()])
        period = period_map.get(base_freq, 7) # default 7 if mapped frequency not found exactly but inferred
        logger.log_warning(f"Inferred frequency '{freq}' mapped to period {period}.")
        
    target_cols = columns if columns else [c for c in df.columns if c != date_col]
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(ts_df[c])]
    
    output_data: dict[str, Any] = {}
    output_plots: list[matplotlib.figure.Figure] = []
    
    for col in numeric_cols:
        s = ts_df[col].dropna()
        n = len(s)
        if n < period * 2:
            logger.log_warning(f"Time series '{col}' is too short for decomposition (N={n} < 2*period={period*2}).")
            continue
            
        try:
            res = seasonal_decompose(s, model=model, period=period)
        except Exception as e:
            logger.log_warning(f"Decomposition failed for '{col}': {e}")
            continue
            
        trend = res.trend.dropna()
        seasonal = res.seasonal.dropna()
        resid = res.resid.dropna()
        
        t_str = float(1 - (resid.var() / (resid + trend).var())) if len(resid) > 1 and len(trend) > 1 else 0.0
        s_str = float(1 - (resid.var() / (resid + seasonal).var())) if len(resid) > 1 and len(seasonal) > 1 else 0.0
        
        output_data[col] = {
            'trend_strength': max(0.0, min(1.0, t_str)),
            'seasonal_strength': max(0.0, min(1.0, s_str)),
            'residual_std': float(resid.std())
        }
        
        fig, axes = plot_utils.create_figure_grid(title=f"Decomposition ({model}): {col}", nrows=4, ncols=1)
        
        axes[0].plot(s.index, s.values, color=plot_utils.DEFAULT_COLOR)
        axes[0].set_ylabel('Original')
        
        axes[1].plot(trend.index, trend.values, color=plot_utils.DEFAULT_COLOR)
        axes[1].set_ylabel('Trend')
        
        axes[2].plot(seasonal.index, seasonal.values, color=plot_utils.DEFAULT_COLOR)
        axes[2].set_ylabel('Seasonal')
        
        axes[3].plot(resid.index, resid.values, 'o', color=plot_utils.DEFAULT_COLOR, markersize=3)
        axes[3].axhline(0, color='r', linestyle='--')
        axes[3].set_ylabel('Residual')
        
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    return output_data, output_plots
