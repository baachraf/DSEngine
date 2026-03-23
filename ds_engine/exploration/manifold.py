"""
ds_engine.exploration.manifold
==============================

Perform dimensionality reduction and plot the resulting manifold.
"""

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.figure
from typing import Any

from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Reduce dataset dimensions to 2D and visualize.
    
    Args:
        df: Input DataFrame.
        columns: Numeric columns to reduce.
        params: Config params including method (pca, tsne) and target_column.
        
    Returns:
        tuple[dict, list]: Result parameters and reduction scatter plots.
    """
    method = params.get('method', 'pca').lower()
    target_column = params.get('target_column', None)
    
    target_cols = columns if columns else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if target_column and target_column in target_cols:
        target_cols.remove(target_column)
        
    if len(target_cols) < 3:
        logger.log_warning("Dimensionality reduction is generally used for >3 dimensions.")
        
    analysis_df = df[target_cols].dropna()
    if len(analysis_df) < 5:
        return {'status': 'skipped', 'reason': 'insufficient data'}, []
        
    output_data: dict[str, Any] = {'method': method, 'original_dimensions': len(target_cols)}
    output_plots: list[matplotlib.figure.Figure] = []
    
    plot_df = pd.DataFrame(index=analysis_df.index)
    if target_column and target_column in df.columns:
        plot_df['Target'] = df.loc[analysis_df.index, target_column]
    else:
        plot_df['Target'] = 'All Data'
        
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42)
        embedded = reducer.fit_transform(analysis_df)
        title = "t-SNE 2D Projection"
    else:
        # Defaults to PCA
        reducer = PCA(n_components=2, random_state=42)
        embedded = reducer.fit_transform(analysis_df)
        output_data['explained_variance_ratio'] = [float(x) for x in reducer.explained_variance_ratio_]
        output_data['total_variance_explained'] = float(sum(reducer.explained_variance_ratio_))
        title = f"PCA 2D Projection ({output_data['total_variance_explained']*100:.1f}% Variance)"
        
    plot_df['Component 1'] = embedded[:, 0]
    plot_df['Component 2'] = embedded[:, 1]
    
    fig, ax = plot_utils.create_figure(title=title)
    
    is_categorical = len(plot_df['Target'].unique()) <= 10 or not pd.api.types.is_numeric_dtype(plot_df['Target'])
    if is_categorical:
        sns.scatterplot(data=plot_df, x='Component 1', y='Component 2', hue='Target', alpha=0.7, ax=ax)
    else:
        scatter = ax.scatter(plot_df['Component 1'], plot_df['Component 2'], c=plot_df['Target'], cmap='viridis', alpha=0.7)
        fig.colorbar(scatter, ax=ax, label=target_column)
        
    plot_utils.finalize_figure(fig)
    output_plots.append(fig)
    
    return output_data, output_plots
