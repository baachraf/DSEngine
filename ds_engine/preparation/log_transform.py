"""
ds_engine.preparation.log_transform
=====================================

Apply a manual monotonic transformation (log, log1p, sqrt) to numeric columns.
Useful for compressing right-skewed distributions without needing sklearn.

This module follows the preparation block contract:
    run(df, columns, params) -> (output_data, output_plots, df_modified)

Expected params keys:
    method (str): 'log1p' (default), 'log', or 'sqrt'.
                  'log'   — natural log; requires all values > 0.
                  'log1p' — natural log(1 + x); requires all values >= 0.
                  'sqrt'  — square root; requires all values >= 0.
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
    """Apply a log or square-root transform to numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Numeric columns to transform. Empty means all numeric.
        params (dict): Step configuration.

    Returns:
        tuple[dict, list, pd.DataFrame]: Stats, plots, and the modified DataFrame.
    """
    method = params.get('method', 'log1p')
    if method not in ('log', 'log1p', 'sqrt'):
        raise ValueError(f"Unknown log_transform method: '{method}'. Use 'log', 'log1p', or 'sqrt'.")

    df_new = df.copy()
    target_cols = columns if columns else [
        c for c in df_new.columns if pd.api.types.is_numeric_dtype(df_new[c])
    ]

    output_data = {
        'method': method,
        'transformed_columns': {},
    }

    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(df_new[col]):
            logger.log_warning(f"Column '{col}' is not numeric — skipping log transform.")
            continue

        s = df_new[col].dropna()
        if len(s) == 0:
            continue

        # Guard: enforce domain requirements
        if method == 'log' and (s <= 0).any():
            logger.log_warning(
                f"Column '{col}' has non-positive values; 'log' requires all > 0. Skipping."
            )
            continue
        if method in ('log1p', 'sqrt') and (s < 0).any():
            logger.log_warning(
                f"Column '{col}' has negative values; '{method}' requires all >= 0. Skipping."
            )
            continue

        skew_before = float(s.skew())

        if method == 'log':
            df_new[col] = np.log(df_new[col])
        elif method == 'log1p':
            df_new[col] = np.log1p(df_new[col])
        elif method == 'sqrt':
            df_new[col] = np.sqrt(df_new[col])

        skew_after = float(df_new[col].dropna().skew())
        output_data['transformed_columns'][col] = {
            'skew_before': float(round(skew_before, 4)),
            'skew_after':  float(round(skew_after, 4)),
        }

    return output_data, [], df_new
