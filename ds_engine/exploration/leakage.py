"""
ds_engine.exploration.leakage
=============================

Detect potential target leakage features using lightweight trees.
"""

import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.figure
from typing import Any

from ds_engine.utils import plot_utils, logger

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Determine if any independent feature predicts the target too well.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to check against the target.
        params: target_column (required), threshold (default 0.95).
        
    Returns:
        tuple[dict, list]: Suspicious features and a barplot visualization.
    """
    target = params.get('target_column')
    threshold = params.get('threshold', 0.95)
    
    if not target or target not in df.columns:
        raise ValueError(f"leakage block requires a valid 'target_column' param. Got {target}")
        
    y = df[target]
    is_categorical = not pd.api.types.is_numeric_dtype(y) or y.nunique() < 10
    
    target_cols = columns if columns else [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c != target]
    
    output_data: dict[str, Any] = {'target': target, 'suspicious_features': []}
    scores = []
    
    for col in target_cols:
        temp = df[[col, target]].dropna()
        if len(temp) < 20:
            continue
            
        X_temp = temp[[col]]
        y_temp = temp[target].astype(str) if is_categorical else temp[target]
        
        if is_categorical:
            model = ExtraTreesClassifier(n_estimators=20, max_depth=5, random_state=42)
            cv_scores = cross_val_score(model, X_temp, y_temp, cv=3, scoring='accuracy')
        else:
            model = ExtraTreesRegressor(n_estimators=20, max_depth=5, random_state=42)
            cv_scores = cross_val_score(model, X_temp, y_temp, cv=3, scoring='r2')
            
        mean_score = max(0.0, float(cv_scores.mean()))
        scores.append({'feature': col, 'score': mean_score})
        
        if mean_score > threshold:
            output_data['suspicious_features'].append({
                'feature': col, 
                'score': mean_score, 
                'reason': f"{'Accuracy' if is_categorical else 'R2'} > {threshold}"
            })
            
    output_plots: list[matplotlib.figure.Figure] = []
    
    if scores:
        plot_df = pd.DataFrame(scores).sort_values('score', ascending=False).head(15)
        fig, ax = plot_utils.create_figure(title=f"Target Leakage Check -> {target}")
        
        # Highlight leaks in red
        colors = ['#e74c3c' if s > threshold else plot_utils.DEFAULT_COLOR for s in plot_df['score']]
        sns.barplot(data=plot_df, x='score', y='feature', ax=ax, palette=colors)
        ax.axvline(x=threshold, color='k', linestyle='--', label=f'Leak Threshold ({threshold})')
        ax.set_xlabel("Predictive Confidence (Score)")
        ax.set_ylabel("Single Feature")
        ax.legend()
        
        plot_utils.finalize_figure(fig)
        output_plots.append(fig)
        
    return output_data, output_plots
