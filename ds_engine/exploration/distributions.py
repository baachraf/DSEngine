"""
ds_engine.exploration.distributions
===================================

Analyze and visualize the statistical distribution of numeric columns.
This module follows the standard block contract.

Expected params keys:
    plot_type (str): 'histogram', 'kde', or 'both'. Default 'both'.
    bins (int): Default 30.
    show_normal_overlay (bool): Default False.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import seaborn as sns
import matplotlib.figure
from typing import Any
from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Visualize and quantify the distribution of each specified column.
    
    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Columns to analyze. Empty means all numeric.
        params (dict): Step configuration.
        
    Returns:
        tuple[dict, list]: Per-column stats and a list of Figures.
    """
    plot_type = params.get('plot_type', 'both')
    bins = params.get('bins', 30)
    show_normal = params.get('show_normal_overlay', False)
    
    if plot_type not in ['histogram', 'kde', 'both']:
        raise ValueError(f"plot_type must be 'histogram', 'kde', or 'both'. Got '{plot_type}'.")
        
    target_cols = columns if columns else df.columns
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    skipped = set(target_cols) - set(numeric_cols)
    if skipped:
        logger.log_warning(f"distributions skipped non-numeric columns: {list(skipped)}")
        
    output_data: dict[str, dict[str, float]] = {}
    output_plots: list[matplotlib.figure.Figure] = []
    
    for col in numeric_cols:
        valid_s = df[col].dropna()
        n_valid = len(valid_s)
        
        # Stats
        col_stats: dict[str, float] = {}
        if n_valid > 0:
            col_stats['mean'] = float(valid_s.mean())
            col_stats['std'] = float(valid_s.std())
            if n_valid > 2:
                col_stats['skewness'] = float(valid_s.skew())
            if n_valid > 3:
                col_stats['kurtosis'] = float(valid_s.kurtosis())
            
            # Normality p-value (Shapiro-Wilk)
            if 3 <= n_valid <= 5000:
                _, p_val = stats.shapiro(valid_s)
                col_stats['normality_p_value'] = float(p_val)
            elif n_valid > 5000:
                # Shapiro is unreliable > 5000, fallback to D'Agostino's K-squared
                _, p_val = stats.normaltest(valid_s)
                col_stats['normality_p_value'] = float(p_val)
                
        output_data[col] = col_stats
        
        # Plots
        if n_valid > 0:
            fig, ax = plot_utils.create_figure(title=f"Distribution: {col}")
            
            if plot_type == 'histogram':
                sns.histplot(valid_s, bins=bins, kde=False, color=plot_utils.DEFAULT_COLOR, ax=ax)
            elif plot_type == 'kde':
                sns.kdeplot(valid_s, color=plot_utils.DEFAULT_COLOR, ax=ax, lw=2)
            else: # both
                sns.histplot(valid_s, bins=bins, kde=True, color=plot_utils.DEFAULT_COLOR, ax=ax)
                
            if show_normal and col_stats.get('std', 0) > 0:
                # Overlay normal curve
                mu, std = col_stats['mean'], col_stats['std']
                xmin, xmax = ax.get_xlim()
                x = np.linspace(xmin, xmax, 100)
                p = stats.norm.pdf(x, mu, std)
                # Scale PDF to match histogram area or KDE depending on plot_type
                if plot_type == 'histogram':
                    # Scale to match counts
                    area = len(valid_s) * ((xmax - xmin) / bins)
                    p = p * area
                
                ax.plot(x, p, 'k--', linewidth=2, label='Normal Dist')
                ax.legend()
                
            plot_utils.finalize_figure(fig)
            output_plots.append(fig)
            
    return output_data, output_plots
