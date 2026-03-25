"""
ds_engine.preparation.bin
==========================

Discretise (bin) numeric columns into labelled categories.
Supports equal-width (uniform) and quantile-based (quantile) strategies.

This module follows the preparation block contract:
    run(df, columns, params) -> (output_data, output_plots, df_modified)

Expected params keys:
    method (str): 'uniform' (equal-width, default) or 'quantile'.
    n_bins (int): Number of bins. Default 5.
    labels (list[str] | None): Optional bin labels. If provided, must equal n_bins.
    suffix (str): Suffix appended to the new column name. Default '_bin'.
    drop_original (bool): Drop the original numeric column. Default False.
"""

import pandas as pd
import matplotlib.figure
from typing import Any
from ds_engine.utils import logger


def run(
    df: pd.DataFrame,
    columns: list[str],
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure], pd.DataFrame]:
    """Bin numeric columns into categorical intervals.

    Args:
        df (pd.DataFrame): Input DataFrame. Never modified in-place.
        columns (list[str]): Numeric columns to bin. Empty means all numeric.
        params (dict): Step configuration.

    Returns:
        tuple[dict, list, pd.DataFrame]: Stats, plots, and the modified DataFrame.
    """
    method        = params.get('method', 'uniform')
    n_bins        = int(params.get('n_bins', 5))
    labels        = params.get('labels', None)
    suffix        = params.get('suffix', '_bin')
    drop_original = bool(params.get('drop_original', False))

    if method not in ('uniform', 'quantile'):
        raise ValueError(f"Unknown bin method: '{method}'. Use 'uniform' or 'quantile'.")

    if labels is not None and len(labels) != n_bins:
        raise ValueError(
            f"'labels' length ({len(labels)}) must equal n_bins ({n_bins})."
        )

    df_new = df.copy()
    target_cols = columns if columns else [
        c for c in df_new.columns if pd.api.types.is_numeric_dtype(df_new[c])
    ]

    output_data = {
        'method': method,
        'n_bins': n_bins,
        'binned_columns': {},
    }

    for col in target_cols:
        if not pd.api.types.is_numeric_dtype(df_new[col]):
            logger.log_warning(f"Column '{col}' is not numeric — skipping binning.")
            continue

        s = df_new[col].dropna()
        if len(s) == 0:
            continue

        new_col = col + suffix
        try:
            if method == 'uniform':
                df_new[new_col] = pd.cut(df_new[col], bins=n_bins, labels=labels)
            else:  # quantile
                df_new[new_col] = pd.qcut(df_new[col], q=n_bins, labels=labels, duplicates='drop')
        except Exception as e:
            logger.log_warning(f"Could not bin column '{col}': {e}")
            continue

        bin_counts = df_new[new_col].value_counts(sort=False).to_dict()
        output_data['binned_columns'][col] = {
            'new_column': new_col,
            'bin_counts': {str(k): int(v) for k, v in bin_counts.items()},
        }

        if drop_original:
            df_new.drop(columns=[col], inplace=True)

    return output_data, [], df_new
