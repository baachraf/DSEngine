"""
ds_engine.exploration.missing
=============================

Analyze and visualize missing values across specified columns.
This module follows the standard block contract.

Expected params keys:
    plot_type (str): 'bar', 'heatmap', or 'both'. Default 'bar'.
"""

import pandas as pd
import seaborn as sns
import matplotlib.figure
from typing import Any
from ds_engine.utils import plot_utils

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    target_cols = columns if columns else df.columns
    sub_df = df[target_cols]
    
    plot_type = params.get('plot_type', 'bar')
    n_rows = len(sub_df)
    
    total_missing = int(sub_df.isna().sum().sum())
    total_cells = int(sub_df.shape[0] * sub_df.shape[1])
    
    output_data: dict[str, Any] = {
        'total_missing': total_missing,
        'total_missing_pct': float((total_missing / total_cells) * 100) if total_cells > 0 else 0.0,
        'per_column': {},
        'columns_with_no_missing': [],
        'columns_fully_missing': []
    }
    
    missing_counts = sub_df.isna().sum()
    
    for col in target_cols:
        count = int(missing_counts[col])
        pct = float((count / n_rows) * 100) if n_rows > 0 else 0.0
        
        output_data['per_column'][col] = {
            'null_count': count,
            'null_pct': pct
        }
        
        if count == 0:
            output_data['columns_with_no_missing'].append(col)
        elif count == n_rows:
            output_data['columns_fully_missing'].append(col)
            
    output_plots: list[matplotlib.figure.Figure] = []
    
    if plot_type in ['bar', 'both']:
        fig, ax = plot_utils.create_figure(title="Missing Values % by Column")
        # Filter to columns that actually have missing to avoid noise, unless all have 0
        missing_pcts = pd.Series({c: output_data['per_column'][c]['null_pct'] for c in target_cols})
        
        # Plot all columns to show completeness context
        missing_pcts.sort_values(ascending=False).plot(kind='bar', ax=ax, color=plot_utils.DEFAULT_COLOR)
        ax.set_ylabel('% Missing')
        ax.set_ylim(0, 100)
        
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    if plot_type in ['heatmap', 'both'] and total_missing > 0:
        fig, ax = plot_utils.create_figure(title="Missing Values Map", figsize=plot_utils.FIGSIZE_WIDE)
        cmap = sns.color_palette(["#fcfcfc", plot_utils.DEFAULT_COLOR])
        
        # Downsample rows for heatmap if huge to save memory
        sample_df = sub_df if n_rows <= 10000 else sub_df.sample(str(10000), random_state=42)
        
        sns.heatmap(sample_df.isna(), cmap=cmap, cbar=False, yticklabels=False, ax=ax)
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    return output_data, output_plots
