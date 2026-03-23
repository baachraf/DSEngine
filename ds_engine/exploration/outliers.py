"""
ds_engine.exploration.outliers
==============================

Detect and visualize outliers in numeric columns.
This module follows the standard block contract.

Expected params keys:
    method (str): 'iqr' or 'zscore'. Default 'iqr'.
    threshold (float): IQR multiplier or Z-score cutoff. Default 1.5 for iqr, 3.0 for zscore.
    plot_type (str): 'boxplot' or 'scatter'. Default 'boxplot'.
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.figure
from typing import Any
from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    method = params.get('method', 'iqr')
    default_thresh = 1.5 if method == 'iqr' else 3.0
    threshold = params.get('threshold', default_thresh)
    plot_type = params.get('plot_type', 'boxplot')
    
    target_cols = columns if columns else df.columns
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    skipped = set(target_cols) - set(numeric_cols)
    if skipped:
        logger.log_warning(f"outliers skipped non-numeric columns: {list(skipped)}")

    output_data: dict[str, Any] = {}
    output_plots: list[matplotlib.figure.Figure] = []
    
    for col in numeric_cols:
        s = df[col]
        valid_s = s.dropna()
        n_valid = len(valid_s)
        
        if n_valid == 0:
            continue
            
        lower_bound = None
        upper_bound = None
        outlier_mask = pd.Series(False, index=s.index)
        
        if method == 'iqr':
            q1 = valid_s.quantile(0.25)
            q3 = valid_s.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - (threshold * iqr)
            upper_bound = q3 + (threshold * iqr)
            outlier_mask = (s < lower_bound) | (s > upper_bound)
            
        elif method == 'zscore':
            mean = valid_s.mean()
            std = valid_s.std()
            if std > 0:
                lower_bound = mean - (threshold * std)
                upper_bound = mean + (threshold * std)
                outlier_mask = (s < lower_bound) | (s > upper_bound)
                
        outlier_idx = s[outlier_mask].index.tolist()
        count = len(outlier_idx)
        pct = float((count / n_valid) * 100) if n_valid > 0 else 0.0
        
        output_data[col] = {
            'outlier_count': count,
            'outlier_pct': pct,
            'outlier_indices': [int(i) for i in outlier_idx],
            'lower_bound': float(lower_bound) if lower_bound is not None else None,
            'upper_bound': float(upper_bound) if upper_bound is not None else None
        }
        
        fig, ax = plot_utils.create_figure(title=f"Outliers: {col}")
        
        if plot_type == 'boxplot':
            sns.boxplot(y=valid_s, ax=ax, color=plot_utils.DEFAULT_COLOR)
        else: # scatter
            x_range = range(n_valid)
            ax.scatter(x_range, valid_s, c=plot_utils.get_palette(2)[0], alpha=0.5)
            # highlight outliers
            out_s = valid_s[valid_s.index.isin(outlier_idx)]
            ax.scatter(np.where(valid_s.index.isin(outlier_idx))[0], out_s, c='red')
            
            if lower_bound is not None:
                ax.axhline(lower_bound, color='red', linestyle='--')
            if upper_bound is not None:
                ax.axhline(upper_bound, color='red', linestyle='--')
                
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    return output_data, output_plots
