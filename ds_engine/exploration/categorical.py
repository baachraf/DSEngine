"""
ds_engine.exploration.categorical
=================================

Analyze and visualize the distribution of categorical columns.
This module follows the standard block contract.

Expected params keys:
    top_n (int): Show top N categories. Default 10.
    plot_type (str): 'bar' or 'pie'. Default 'bar'.
    show_percentages (bool): Default True.
"""

import pandas as pd
import seaborn as sns
import matplotlib.figure
from typing import Any
from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    top_n = params.get('top_n', 10)
    plot_type = params.get('plot_type', 'bar')
    show_percentages = params.get('show_percentages', True)
    
    target_cols = columns if columns else df.columns
    cat_cols = [c for c in target_cols if pd.api.types.is_object_dtype(df[c]) or isinstance(df[c].dtype, pd.CategoricalDtype)]
    
    skipped = set(target_cols) - set(cat_cols)
    if skipped:
        logger.log_warning(f"categorical skipped non-object columns: {list(skipped)}")

    output_data: dict[str, Any] = {}
    output_plots: list[matplotlib.figure.Figure] = []
    
    for col in cat_cols:
        s = df[col].dropna()
        n_valid = len(s)
        if n_valid == 0:
            continue
            
        counts = s.value_counts()
        unique_cnt = len(counts)
        pcts = s.value_counts(normalize=True) * 100
        
        top_cats = []
        for val in counts.head(top_n).index:
            top_cats.append({
                'value': str(val),
                'count': int(counts[val]),
                'pct': float(pcts[val])
            })
            
        rare_cnt = int((pcts < 1.0).sum())
        
        output_data[col] = {
            'unique_count': int(unique_cnt),
            'top_n_categories': top_cats,
            'rare_categories_count': rare_cnt
        }
        
        fig, ax = plot_utils.create_figure(title=f"Category Frequencies: {col}")
        top_df = pd.DataFrame(top_cats)
        
        if top_df.empty:
            plt.close(fig) # Cleanup empty fig
            continue
            
        if plot_type == 'pie':
            labels = top_df['value'].tolist()
            sizes = top_df['count'].tolist()
            
            # If there's a long tail, lump into "Other"
            if unique_cnt > top_n:
                other_cnt = n_valid - sum(sizes)
                if other_cnt > 0:
                    labels.append('Other')
                    sizes.append(other_cnt)
                    
            autopct = '%1.1f%%' if show_percentages else None
            ax.pie(sizes, labels=labels, autopct=autopct, startangle=90, 
                   colors=plot_utils.get_palette(len(sizes)))
        else: # bar
            sns.barplot(data=top_df, x='count', y='value', color=plot_utils.DEFAULT_COLOR, ax=ax)
            ax.set_xlabel('Count')
            ax.set_ylabel('')
            
            if show_percentages:
                for idx, row in top_df.iterrows():
                    ax.text(row['count'], idx, f" {row['pct']:.1f}%", va='center')
                    
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    return output_data, output_plots
