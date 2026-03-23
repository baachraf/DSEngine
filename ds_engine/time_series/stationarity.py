"""
ds_engine.time_series.stationarity
==================================

Test whether a time series column is stationary using ADF and KPSS tests.
This module follows the standard block contract.

Expected params keys:
    date_column (str): REQUIRED. Name of datetime column for correct ordering.
    alpha (float): Significance level. Default 0.05.
"""

import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss
import matplotlib.figure
from typing import Any
from ds_engine.utils import logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    date_col = params.get('date_column')
    if not date_col:
        raise ValueError("stationarity requires 'date_column' parameter.")
    if date_col not in df.columns:
        raise KeyError(f"date_column '{date_col}' not found in DataFrame.")
        
    alpha = float(params.get('alpha', 0.05))
    
    # Sort values safely
    ts_df = df.sort_values(date_col)
    
    target_cols = columns if columns else [c for c in df.columns if c != date_col]
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(ts_df[c])]
    
    output_data: dict[str, Any] = {}
    
    for col in numeric_cols:
        s = ts_df[col].dropna()
        n = len(s)
        if n < 10:
            logger.log_warning(f"Series '{col}' is too short for stationarity testing.")
            continue
            
        adf_stat, adf_p = None, None
        try:
            adf_res = adfuller(s)
            adf_stat = float(adf_res[0])
            adf_p = float(adf_res[1])
        except Exception as e:
            logger.log_warning(f"ADF test failed for '{col}': {e}")
            
        kpss_stat, kpss_p = None, None
        try:
            # nlags="auto" requires more recent statsmodels, fallback implemented implicitly by default
            kpss_res = kpss(s, regression='c', nlags='auto')
            kpss_stat = float(kpss_res[0])
            kpss_p = float(kpss_res[1])
        except Exception as e:
            logger.log_warning(f"KPSS test failed for '{col}': {e}")
            
        if adf_p is not None and kpss_p is not None:
            # ADF: H0 = non-stationary (p < alpha => stationary)
            adf_stationary = (adf_p < alpha)
            
            # KPSS: H0 = stationary (p < alpha => non-stationary)
            kpss_stationary = (kpss_p >= alpha)
            
            is_stat = adf_stationary and kpss_stationary
            
            interp = "Strictly Stationary" if is_stat else "Non-stationary or Mixed evidence"
            
            if adf_stationary and not kpss_stationary:
                interp = "Difference stationary (unit root but trend)"
            elif not adf_stationary and kpss_stationary:
                interp = "Trend stationary (no unit root but deterministic trend)"
                
            output_data[col] = {
                'adf_statistic': adf_stat,
                'adf_p_value': adf_p,
                'kpss_statistic': kpss_stat,
                'kpss_p_value': kpss_p,
                'is_stationary': bool(is_stat),
                'interpretation': interp
            }
            
    return output_data, []  # No plots for stationarity 
