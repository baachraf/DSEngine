"""
ds_engine.exploration.multivariate_outliers
===========================================

Detect multidimensional anomalies using Isolation Forest algorithm.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.figure
from typing import Any

from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Detect multivariate outliers using Isolation Forest.
    
    Args:
        df: Input DataFrame.
        columns: Numeric columns to use for modeling context.
        params: Configuration dictionary (contamination).
        
    Returns:
        tuple[dict, list]: Result parameters and a PCA 2D plotted outlier visual.
    """
    contamination = params.get('contamination', 'auto')
    
    target_cols = columns if columns else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    
    if len(target_cols) < 2:
        logger.log_warning("Multivariate outlier detection requires at least 2 numeric columns.")
        return {'status': 'skipped', 'reason': 'insufficient numeric columns'}, []
        
    analysis_df = df[target_cols].dropna()
    if len(analysis_df) < 20:
        logger.log_warning("Too few non-null rows for Isolation Forest (<20).")
        return {'status': 'skipped', 'reason': 'insufficient data'}, []
        
    # Fit IF
    model = IsolationForest(contamination=contamination, random_state=42)
    preds = model.fit_predict(analysis_df)
    
    # Evaluate Results
    is_outlier = preds == -1
    anomaly_scores = model.decision_function(analysis_df)
    
    # Store Output
    outlier_indices = analysis_df[is_outlier].index.tolist()
    total_outliers = int(is_outlier.sum())
    
    output_data = {
        'total_analyzed': len(analysis_df),
        'total_outliers': total_outliers,
        'outlier_percentage': float(total_outliers / len(analysis_df) * 100),
        'outlier_indices': outlier_indices[:100],  # cap at 100 for JSON safety
        'contamination_used': str(contamination)
    }
    
    output_plots: list[matplotlib.figure.Figure] = []
    
    # Generate PCA Projection Plot to visualize the outliers in 2D space
    if len(analysis_df) > 0 and len(target_cols) >= 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(analysis_df)
        plot_df = pd.DataFrame(pca_result, columns=['PC1', 'PC2'])
        plot_df['Outlier'] = ['Anomaly' if x else 'Normal' for x in is_outlier]
        
        fig, ax = plot_utils.create_figure(title="Multivariate Outliers (PCA Projection)")
        sns.scatterplot(
            data=plot_df, x='PC1', y='PC2', hue='Outlier', 
            palette={'Normal': plot_utils.DEFAULT_COLOR, 'Anomaly': '#e74c3c'},
            alpha=0.7, ax=ax
        )
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    return output_data, output_plots
