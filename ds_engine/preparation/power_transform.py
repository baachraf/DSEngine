"""
ds_engine.preparation.power_transform
======================================

Apply Yeo-Johnson or Box-Cox power transformation to numeric columns.
Reduces skewness and pulls distributions toward normality.

This module follows the preparation block contract:
    run(df, columns, params) -> (output_data, output_plots, df_modified)

Expected params keys:
    method (str): 'yeo-johnson' (default) or 'box-cox'.
                  Note: box-cox requires all values > 0.
"""

import numpy as np
import pandas as pd
import matplotlib.figure
from typing import Any
from sklearn.preprocessing import PowerTransformer
from ds_engine.utils import logger


def run(
    df: pd.DataFrame,
    columns: list[str],
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure], pd.DataFrame]:
    """Apply a power transformation (Yeo-Johnson or Box-Cox) to numeric columns.

    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Numeric columns to transform. Empty means all numeric.
        params (dict): Step configuration.

    Returns:
        tuple[dict, list, pd.DataFrame]: Stats, plots, and the modified DataFrame.
    """
    method = params.get('method', 'yeo-johnson')
    if method not in ('yeo-johnson', 'box-cox'):
        raise ValueError(f"Unknown power_transform method: '{method}'. Use 'yeo-johnson' or 'box-cox'.")

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
            logger.log_warning(f"Column '{col}' is not numeric — skipping power transform.")
            continue

        s = df_new[col].dropna()
        if len(s) == 0:
            continue

        if method == 'box-cox' and (s <= 0).any():
            logger.log_warning(
                f"Column '{col}' contains non-positive values; "
                f"Box-Cox requires all values > 0. Skipping."
            )
            continue

        skew_before = float(s.skew())

        # Fit + transform only the non-null rows
        pt = PowerTransformer(method=method, standardize=False)
        valid_mask = df_new[col].notna()
        vals = df_new.loc[valid_mask, col].values.reshape(-1, 1)
        
        # Cast to float to prevent LossySetitemError when assigning float to int col
        df_new[col] = df_new[col].astype(float)
        df_new.loc[valid_mask, col] = pt.fit_transform(vals).ravel()

        skew_after = float(df_new[col].dropna().skew())
        output_data['transformed_columns'][col] = {
            'lambda': float(round(pt.lambdas_[0], 4)),
            'skew_before': float(round(skew_before, 4)),
            'skew_after':  float(round(skew_after, 4)),
        }

    return output_data, [], df_new
