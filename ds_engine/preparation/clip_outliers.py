"""
ds_engine.preparation.clip_outliers
=====================================

Clip (winsorise) extreme values in numeric columns.
Prevents a small number of outliers from dominating scale-sensitive
algorithms without entirely removing rows.

This module follows the preparation block contract:
    run(df, columns, params) -> (output_data, output_plots, df_modified)

Expected params keys:
    method (str): 'iqr' (default) or 'zscore'.
    factor (float): IQR multiplier (default 1.5) OR Z-score threshold (default 3.0).
"""

import numpy as np
import pandas as pd
import matplotlib.figure
from typing import Any
from ds_engine.utils import logger


def run(
    df: pd.DataFrame,
    columns: list[str],
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure], pd.DataFrame]:
    """Clip outliers in numeric columns using IQR or Z-score capping.

    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Numeric columns to process. Empty means all numeric.
        params (dict): Step configuration.

    Returns:
        tuple[dict, list, pd.DataFrame]: Stats, plots, and the modified DataFrame.
    """
    method = params.get('method', 'iqr')
    factor = float(params.get('factor', 1.5 if method == 'iqr' else 3.0))

    if method not in ('iqr', 'zscore'):
        raise ValueError(f"Unknown clip_outliers method: '{method}'. Use 'iqr' or 'zscore'.")

    df_new = df.copy()
    target_cols = columns if columns else [
        c for c in df_new.columns if pd.api.types.is_numeric_dtype(df_new[c])
    ]

    output_data = {
        'method': method,
        'factor': factor,
        'clipped_columns': {},
    }

    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(df_new[col]):
            logger.log_warning(f"Column '{col}' is not numeric — skipping outlier clipping.")
            continue

        s = df_new[col].dropna()
        if len(s) == 0:
            continue

        if method == 'iqr':
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - factor * iqr
            upper = q3 + factor * iqr
        else:  # zscore
            mu, std = s.mean(), s.std()
            lower = mu - factor * std
            upper = mu + factor * std

        n_clipped_low  = int((df_new[col] < lower).sum())
        n_clipped_high = int((df_new[col] > upper).sum())

        df_new[col] = df_new[col].clip(lower=lower, upper=upper)

        output_data['clipped_columns'][col] = {
            'lower_bound': float(round(lower, 6)),
            'upper_bound': float(round(upper, 6)),
            'n_clipped_low':  n_clipped_low,
            'n_clipped_high': n_clipped_high,
            'total_clipped':  n_clipped_low + n_clipped_high,
        }

    return output_data, [], df_new
