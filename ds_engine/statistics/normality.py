"""
ds_engine.statistics.normality
==============================

Test whether each specified column follows a normal distribution.
This module follows the standard block contract.

Expected params keys:
    test (str): 'shapiro', 'dagostino', or 'ks'. Default 'shapiro'.
    alpha (float): Significance level. Default 0.05.
    plot_qq (bool): Add Q-Q plot per column. Default True.
"""

import pandas as pd
import scipy.stats as stats
import matplotlib.figure
import statsmodels.api as sm
from typing import Any
from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    test = params.get('test', 'shapiro')
    alpha = float(params.get('alpha', 0.05))
    plot_qq = params.get('plot_qq', True)
    
    target_cols = columns if columns else df.columns
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    skipped = set(target_cols) - set(numeric_cols)
    if skipped:
        logger.log_warning(f"normality skipped non-numeric columns: {list(skipped)}")
        
    output_data: dict[str, Any] = {}
    output_plots: list[matplotlib.figure.Figure] = []
    
    for col in numeric_cols:
        s = df[col].dropna()
        n = len(s)
        if n < 3:
            continue
            
        stat, p_val = None, None
        try:
            if test == 'shapiro' and n <= 5000:
                stat, p_val = stats.shapiro(s)
            elif test == 'dagostino' or (test == 'shapiro' and n > 5000):
                if test == 'shapiro':
                    logger.log_warning(f"Shapiro unreliable for N > 5000 in '{col}'. Falling back to D'Agostino.")
                    test = 'dagostino'
                stat, p_val = stats.normaltest(s)
            elif test == 'ks':
                stat, p_val = stats.kstest(s, 'norm', args=(s.mean(), s.std()))
        except Exception as e:
            logger.log_warning(f"Normality test {test} failed on '{col}': {e}")
            continue
            
        if p_val is not None:
            output_data[col] = {
                'test_name': test,
                'statistic': float(stat),
                'p_value': float(p_val),
                'is_normal': bool(p_val >= alpha)
            }
            
            if plot_qq:
                fig, ax = plot_utils.create_figure(title=f"Q-Q Plot: {col}")
                sm.qqplot(s, line='45', ax=ax, fit=True)
                # Apply our style back to the plot lines
                ax.get_lines()[0].set_color(plot_utils.DEFAULT_COLOR)
                ax.get_lines()[0].set_markersize(4)
                ax.get_lines()[1].set_color('red')
                
                plot_utils.finalize_figure(fig)
                output_plots.append(fig)
                
    return output_data, output_plots
