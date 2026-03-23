"""
ds_engine.exploration.pps
=========================

Calculate a pseudo Predictive Power Score (PPS) between features.
This provides an asymmetric, non-linear alternative to standard correlation.
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
import seaborn as sns
import matplotlib.figure
from typing import Any
from itertools import permutations

from ds_engine.utils import plot_utils, logger

def _compute_pps(df: pd.DataFrame, x_col: str, y_col: str) -> float:
    """Compute asymmetric predictive power of x_col predicting y_col."""
    temp = df[[x_col, y_col]].dropna()
    if len(temp) < 10:
        return 0.0
        
    X = temp[[x_col]]
    y = temp[y_col]
    
    is_categorical = not pd.api.types.is_numeric_dtype(y) or y.nunique() < 10
    
    if is_categorical:
        # For categorical, we encode it
        y = y.astype(str)
        model = DecisionTreeClassifier(max_depth=4, class_weight='balanced')
        # Baseline accuracy
        majority_class_pct = y.value_counts(normalize=True).max()
        scores = cross_val_score(model, X, y, cv=3, scoring='accuracy')
        score = scores.mean()
        # Scale score relative to baseline
        scaled = (score - majority_class_pct) / (1 - majority_class_pct) if majority_class_pct < 1 else 0
        return max(0.0, float(scaled))
    else:
        model = DecisionTreeRegressor(max_depth=4)
        scores = cross_val_score(model, X, y, cv=3, scoring='r2')
        score = scores.mean()
        # R2 is practically 0 to 1, but can be negative. We bound it at 0.
        return max(0.0, float(score))

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    """Generate Predictive Power Scores between columns.
    
    Args:
        df: Input DataFrame.
        columns: List of columns to evaluate. Empty means all numeric/categorical.
        params: Configuration containing `target` or `plot_type`.
        
    Returns:
        tuple[dict, list]: PPS matrix and generated plots.
    """
    target = params.get('target', None)
    target_cols = columns if columns else df.columns.tolist()
    
    output_data: dict[str, Any] = {'matrix': {}, 'top_predictors': []}
    output_plots: list[matplotlib.figure.Figure] = []
    
    # Restrict to predictable column limits to save time
    if len(target_cols) > 20:
        logger.log_warning("PPS matrix limited to first 20 columns for performance.")
        target_cols = target_cols[:20]
        
    # Isolate targets vs predictors
    y_cols = [target] if target and target in df.columns else target_cols
    x_cols = target_cols
    
    matrix = pd.DataFrame(index=y_cols, columns=x_cols, data=0.0)
    
    for y in y_cols:
        for x in x_cols:
            if x == y:
                matrix.loc[y, x] = 1.0
            else:
                score = _compute_pps(df, x, y)
                matrix.loc[y, x] = score
                if target and y == target:
                    output_data['top_predictors'].append({'feature': x, 'pps': score})
                    
    # Save the raw matrix to output
    output_data['matrix'] = matrix.to_dict(orient='index')
    
    # Plotting
    if target:
        output_data['top_predictors'] = sorted(output_data['top_predictors'], key=lambda x: x['pps'], reverse=True)
        # Barplot of features predicting target
        fig, ax = plot_utils.create_figure(title=f"Predictive Power Score for target: {target}")
        plot_df = pd.DataFrame(output_data['top_predictors'])
        if not plot_df.empty:
            sns.barplot(data=plot_df, x='pps', y='feature', ax=ax, color=plot_utils.DEFAULT_COLOR)
            ax.set_ylabel("Predictor")
            ax.set_xlabel("PPS (0 to 1)")
    else:
        # Heatmap
        fig, ax = plot_utils.create_figure(title="Predictive Power Score Matrix", figsize=plot_utils.FIGSIZE_WIDE)
        sns.heatmap(matrix.astype(float), annot=params.get('annotate', True), cmap="Blues", fmt=".2f", 
                    vmin=0, vmax=1, ax=ax)
        ax.set_xlabel("Feature")
        ax.set_ylabel("Target")
        
    plot_utils.finalize_figure(fig)
    output_plots.append(fig)

    return output_data, output_plots
