"""
ds_engine.statistics.drift
==========================

Test for data drift between the current pipeline DataFrame and a reference source.
"""

import sys
import pandas as pd
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.figure
from typing import Any

from ds_engine.data import loader
from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Perform Data Drift detection using KS-tests.
    
    Args:
        df: Input DataFrame (e.g. inference data).
        columns: List of numeric columns to evaluate for drift.
        params: reference_source (required path string) and alpha (default 0.05).
        
    Returns:
        tuple[dict, list]: KS test results per column and visualization.
    """
    ref_source = params.get('reference_source')
    alpha = params.get('alpha', 0.05)
    
    if not ref_source:
        raise ValueError("drift block requires 'reference_source' pointing to a valid reference CSV/Parquet.")
        
    target_cols = columns if columns else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    # Load reference data using the engine's loader!
    # Because config_parser handles pipeline paths, reference_source must be absolute or relative to the script running.
    # The config_parser can't automatically parse inside params dictionaries.
    try:
        ref_df = loader.load(ref_source, target_cols, {})
    except Exception as e:
        logger.log_error(f"Failed to load reference source for drift block: {e}")
        return {'status': 'FAILED', 'error': 'Reference data rejected'}, []
        
    output_data: dict[str, Any] = {'alpha': alpha, 'columns_drifted': []}
    plots_data = []
    
    for col in target_cols:
        if col not in ref_df.columns:
            continue
            
        cur_s = df[col].dropna()
        ref_s = ref_df[col].dropna()
        
        stat, p_val = ks_2samp(ref_s, cur_s)
        
        has_drifted = bool(p_val < alpha)
        
        output_data[col] = {
            'ks_statistic': float(stat),
            'p_value': float(p_val),
            'drift_detected': has_drifted
        }
        
        if has_drifted:
            output_data['columns_drifted'].append(col)
            
        plots_data.append({'feature': col, 'p_value': p_val, 'drift': has_drifted})
        
    output_plots: list[matplotlib.figure.Figure] = []
    
    if plots_data:
        plot_df = pd.DataFrame(plots_data)
        fig, ax = plot_utils.create_figure(title="Data Drift Detection (Kolmogorov-Smirnov P-Values)")
        
        # Color red if drifted (p < alpha)
        colors = ['#e74c3c' if d else plot_utils.DEFAULT_COLOR for d in plot_df['drift']]
        sns.barplot(data=plot_df, x='p_value', y='feature', ax=ax, palette=colors)
        ax.axvline(x=alpha, color='k', linestyle='--', label=f'Alpha Shield ({alpha})')
        ax.set_xlabel("P-Value (Lower means drifted)")
        ax.set_ylabel("Feature")
        ax.legend()
        
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    return output_data, output_plots
