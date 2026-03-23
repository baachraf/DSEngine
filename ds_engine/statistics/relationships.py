"""
ds_engine.statistics.relationships
==================================

Measure the statistical relationship and significance between column pairs.
This module follows the standard block contract.

Expected params keys:
    pairs (list[list[str]]): Pairs to test. Default: all pairs of `columns`.
    alpha (float): Significance. Default 0.05.
    plot_scatter (bool): Plot scatter for each pair. Default True.
"""

import itertools
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.figure
from typing import Any
from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    target_cols = columns if columns else df.columns
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    alpha = float(params.get('alpha', 0.05))
    plot_scatter = params.get('plot_scatter', True)
    pairs = params.get('pairs')
    
    if not pairs:
        if len(numeric_cols) < 2:
            raise ValueError("relationships step requires at least 2 columns or an explicit pairs list.")
        pairs = list(itertools.combinations(numeric_cols, 2))
        
    output_data: dict[str, Any] = {}
    output_plots: list[matplotlib.figure.Figure] = []
    
    for pair in pairs:
        if len(pair) != 2:
            logger.log_warning(f"Invalid pair length {pair}. Skipping.")
            continue
            
        c1, c2 = pair
        if c1 not in df.columns or c2 not in df.columns:
            logger.log_warning(f"Columns for pair {pair} not in DataFrame. Skipping.")
            continue
            
        valid_df = df[[c1, c2]].dropna()
        if len(valid_df) < 3:
            continue
            
        try:
            corr, p_val = stats.pearsonr(valid_df[c1], valid_df[c2])
        except Exception as e:
            logger.log_warning(f"Correlation failed for {pair}: {e}")
            continue
            
        is_sig = p_val < alpha
        es = "Strong" if abs(corr) > 0.7 else ("Moderate" if abs(corr) > 0.3 else "Weak")
        
        pair_key = f"{c1}_vs_{c2}"
        output_data[pair_key] = {
            'col1': c1,
            'col2': c2,
            'correlation': float(corr),
            'p_value': float(p_val),
            'is_significant': bool(is_sig),
            'effect_size': es
        }
        
        if plot_scatter:
            fig, ax = plot_utils.create_figure(title=f"Relationship: {c1} vs {c2}")
            ax.scatter(valid_df[c1], valid_df[c2], c=plot_utils.DEFAULT_COLOR, alpha=0.5)
            ax.set_xlabel(c1)
            ax.set_ylabel(c2)
            
            # Add trendline
            if valid_df[c1].nunique() > 1:
                z = np.polyfit(valid_df[c1], valid_df[c2], 1)
                p = np.poly1d(z)
                x_fit = np.linspace(valid_df[c1].min(), valid_df[c1].max(), 100)
                ax.plot(x_fit, p(x_fit), 'r--', label=f"Trend (r={corr:.2f})")
                ax.legend()
                
            plot_utils.finalize_figure(fig)
            output_plots.append(fig)
            
    return output_data, output_plots
