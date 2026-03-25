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

def _calculate_cramers_v(ct: pd.DataFrame) -> float:
    """Calculate Cramer's V for a contingency table."""
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    phi2 = chi2 / n
    r, k = ct.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1))) if min((kcorr-1), (rcorr-1)) > 0 else 0.0

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    target_cols = columns if columns else df.columns
    alpha = float(params.get('alpha', 0.05))
    plot_scatter = params.get('plot_scatter', True)
    pairs = params.get('pairs')
    
    if not pairs:
        if len(target_cols) < 2:
            raise ValueError("relationships step requires at least 2 columns or an explicit pairs list.")
        pairs = list(itertools.combinations(target_cols, 2))
        
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
            
        pair_key = f"{c1}_vs_{c2}"
        
        # Case 1: Both Numeric
        if pd.api.types.is_numeric_dtype(df[c1]) and pd.api.types.is_numeric_dtype(df[c2]):
            try:
                corr, p_val = stats.pearsonr(valid_df[c1], valid_df[c2])
                is_sig = p_val < alpha
                es = "Strong" if abs(corr) > 0.7 else ("Moderate" if abs(corr) > 0.3 else "Weak")
                
                output_data[pair_key] = {
                    'type': 'numeric-numeric',
                    'metric': 'pearson',
                    'value': float(corr),
                    'p_value': float(p_val),
                    'is_significant': bool(is_sig),
                    'effect_size': es
                }
                
                if plot_scatter:
                    fig, ax = plot_utils.create_figure(title=f"Relationship: {c1} vs {c2}")
                    ax.scatter(valid_df[c1], valid_df[c2], c=plot_utils.DEFAULT_COLOR, alpha=0.5)
                    ax.set_xlabel(c1)
                    ax.set_ylabel(c2)
                    if valid_df[c1].nunique() > 1:
                        z = np.polyfit(valid_df[c1], valid_df[c2], 1)
                        p = np.poly1d(z)
                        x_fit = np.linspace(valid_df[c1].min(), valid_df[c1].max(), 100)
                        ax.plot(x_fit, p(x_fit), 'r--', label=f"Trend (r={corr:.2f})")
                        ax.legend()
                    plot_utils.finalize_figure(fig)
                    output_plots.append(fig)
            except Exception as e:
                logger.log_warning(f"Numeric correlation failed for {pair}: {e}")

        # Case 2: Both Categorical
        elif (pd.api.types.is_object_dtype(df[c1]) or hasattr(df[c1], 'cat')) and \
             (pd.api.types.is_object_dtype(df[c2]) or hasattr(df[c2], 'cat')):
            try:
                ct = pd.crosstab(valid_df[c1], valid_df[c2])
                chi2, p_val, dof, ex = stats.chi2_contingency(ct)
                v = _calculate_cramers_v(ct)
                es = "Strong" if v > 0.5 else ("Moderate" if v > 0.2 else "Weak")
                
                output_data[pair_key] = {
                    'type': 'categorical-categorical',
                    'metric': "Cramer's V",
                    'value': float(v),
                    'p_value': float(p_val),
                    'is_significant': bool(p_val < alpha),
                    'effect_size': es
                }
            except Exception as e:
                logger.log_warning(f"Categorical association failed for {pair}: {e}")
                
    return output_data, output_plots
