"""
ds_engine.exploration.correlations
==================================

Compute and visualize pairwise correlations between numeric columns.
This module follows the standard block contract.

Expected params keys:
    method (str): 'pearson', 'spearman', or 'kendall'. Default 'pearson'.
    threshold (float): Highlight pairs above this abs value. Default 0.7.
    annotate (bool): Show numbers on heatmap. Default True.
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
    target_cols = columns if columns else df.columns
    numeric_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c])]
    
    skipped = set(target_cols) - set(numeric_cols)
    if skipped:
        logger.log_warning(f"correlations skipped non-numeric columns: {list(skipped)}")

    method = params.get('method', 'pearson')
    threshold = params.get('threshold', 0.7)
    annotate = params.get('annotate', True)
    
    if len(numeric_cols) < 2:
        logger.log_warning("correlations requires at least 2 numeric columns. Skipping.")
        return {}, []
        
    sub_df = df[numeric_cols]
    corr_matrix = sub_df.corr(method=method) # type: ignore
    
    output_data: dict[str, Any] = {
        'matrix': {},
        'high_correlation_pairs': []
    }
    
    cols = corr_matrix.columns.tolist()
    
    # Store matrix
    for c1 in cols:
        output_data['matrix'][c1] = {
            c2: float(corr_matrix.loc[c1, c2]) if pd.notna(corr_matrix.loc[c1, c2]) else None 
            for c2 in cols
        }
        
    # Find high correlations (avoid duplicates/self by iterating upper triangle)
    mat_np = corr_matrix.to_numpy()
    rows, cols_idx = np.where((np.abs(mat_np) >= threshold) & (np.abs(mat_np) != 1.0))
    seen = set()
    
    for r, c in zip(rows, cols_idx):
        if r < c: # Upper triangle only
            pair = (cols[r], cols[c])
            if pair not in seen:
                seen.add(pair)
                val = float(mat_np[r, c])
                if not np.isnan(val):
                    output_data['high_correlation_pairs'].append({
                        'col1': pair[0],
                        'col2': pair[1],
                        'correlation': val
                    })
                
    # Heatmap Figure
    fig, ax = plot_utils.create_figure(title=f"Correlations: {method.capitalize()}", figsize=plot_utils.FIGSIZE_WIDE)
    mask = np.triu(np.ones_like(mat_np, dtype=bool))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(
        corr_matrix, 
        mask=mask, 
        cmap=cmap, 
        vmax=1, vmin=-1, 
        center=0,
        annot=annotate, 
        fmt=".2f",
        square=True, 
        linewidths=.5, 
        cbar_kws={"shrink": .5},
        ax=ax
    )
    
    plot_utils.finalize_figure(fig)
    return output_data, [fig]
