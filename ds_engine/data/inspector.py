"""
ds_engine.data.inspector
========================
Pre-flight inspector to run sanity checks on the dataset.
Logs warnings and raises SystemExit on fatal errors.
"""

import pandas as pd
import numpy as np
import sys
from typing import Any

def inspect(df: pd.DataFrame, columns: list[str], experiment_name: str) -> dict[str, Any]:
    """Inspect DataFrame before pipeline execution. 
    Print summary and all warnings. Never modifies df. 
    Returns inspection_report dict saved in report.json.
    
    Args:
        df (pd.DataFrame): DataFrame to inspect.
        columns (list[str]): Columns requested by the user pipeline config.
        experiment_name (str): Name of the experiment.
        
    Returns:
        dict[str, Any]: Detailed inspection report.
    """
    try:
        from ds_engine.utils import logger
        log = logger.get_logger('inspector')
    except ImportError:
        import logging
        log = logging.getLogger('inspector')

    report = {
        'shape': {'rows': int(df.shape[0]), 'columns': int(df.shape[1])},
        'dtypes': {str(k): str(v) for k, v in df.dtypes.items()},
        'null_summary': {},
        'warnings': [],
        'column_flags': {},
        'warning_count': 0,
        'fatal_count': 0
    }
    
    # FATAL 1: DataFrame has 0 rows
    if df.shape[0] == 0:
        log.error("DataFrame has 0 rows after loading.")
        sys.exit(2)
        
    # FATAL 2: No columns remain
    if df.shape[1] == 0:
        log.error("No columns remain in the DataFrame.")
        sys.exit(2)

    # FATAL 3: Requested column does not exist
    if columns:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            log.error(f"Requested columns do not exist in the DataFrame: {missing}")
            sys.exit(2)
            
    # Check dataset limits
    if df.shape[0] < 30:
        report['warnings'].append({
            'level': 'WARNING',
            'check': 'low_row_count',
            'message': 'Dataset has fewer than 30 rows. This may affect the reliability of statistical tests.'
        })
    elif df.shape[0] > 1000000:
        report['warnings'].append({
            'level': 'WARNING',
            'check': 'high_row_count',
            'message': f"Dataset has {df.shape[0]} rows. Loading into memory may be slow or cause MemoryError. Consider using the 'sample' step as the first step in your pipeline to reduce to a manageable size before running EDA."
        })
        
    # Duplicate rows
    dup_count = int(df.duplicated().sum())
    if dup_count > 0:
        dup_pct = (dup_count / df.shape[0]) * 100
        report['warnings'].append({
            'level': 'WARNING',
            'check': 'duplicate_rows',
            'message': f"Dataset contains {dup_count} duplicate rows ({dup_pct:.1f}%)."
        })

    for col in df.columns:
        report['column_flags'][col] = {
            'all_null': False,
            'zero_variance': False,
            'has_inf': False,
            'dtype_mismatch': False,
            'high_cardinality': False
        }
        
        null_count = int(df[col].isna().sum())
        null_pct = (null_count / df.shape[0]) * 100
        report['null_summary'][col] = {'null_count': null_count, 'null_pct': float(null_pct)}
        
        if null_count == df.shape[0]:
            report['column_flags'][col]['all_null'] = True
            report['warnings'].append({
                'level': 'WARNING', 'column': col, 'check': 'all_null',
                'message': f"Column '{col}' is 100% null."
            })
            continue

        if null_pct > 50:
            report['warnings'].append({
                'level': 'WARNING', 'column': col, 'check': 'high_null_pct',
                'message': f"Column '{col}' has {null_pct:.1f}% null values."
            })

        dtype = df[col].dtype
        
        # Numeric checks
        if pd.api.types.is_numeric_dtype(dtype):
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) == 1:
                report['column_flags'][col]['zero_variance'] = True
                report['warnings'].append({
                    'level': 'WARNING', 'column': col, 'check': 'zero_variance',
                    'message': f"Numeric column '{col}' has zero variance (all values are {unique_vals[0]})."
                })
                
            inf_count = int(np.isinf(df[col]).sum())
            if inf_count > 0:
                report['column_flags'][col]['has_inf'] = True
                report['warnings'].append({
                    'level': 'WARNING', 'column': col, 'check': 'has_inf',
                    'message': f"Numeric column '{col}' contains {inf_count} infinite values."
                })
                
            # Skewness
            try:
                skew_val = float(df[col].skew())
                if abs(skew_val) > 10:
                    report['warnings'].append({
                        'level': 'WARNING', 'column': col, 'check': 'high_skew',
                        'message': f"Column '{col}' has extreme skewness ({skew_val:.2f}). Consider log transform."
                    })
            except Exception:
                pass # e.g. not enough non-null values
                
        # Categorical / Object checks
        elif pd.api.types.is_object_dtype(dtype):
            sample_val = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
            
            # Check if it appears numeric
            if sample_val is not None:
                numeric_try = pd.to_numeric(df[col].dropna().head(100), errors='coerce')
                if numeric_try.notna().all():
                    report['column_flags'][col]['dtype_mismatch'] = True
                    report['warnings'].append({
                        'level': 'WARNING', 'column': col, 'check': 'numeric_as_object',
                        'message': f"A numeric column '{col}' is stored as object dtype. Consider pd.to_numeric()."
                    })
                else:
                    # Check if it appears datetime
                    date_try = pd.to_datetime(df[col].dropna().head(10), errors='coerce', format='mixed')
                    if date_try.notna().all():
                        report['column_flags'][col]['dtype_mismatch'] = True
                        report['warnings'].append({
                            'level': 'WARNING', 'column': col, 'check': 'date_as_object',
                            'message': f"A date column '{col}' is stored as object dtype. Consider pd.to_datetime(). Sample: {sample_val}"
                        })
                        
            # Cardinality
            unique_count = df[col].nunique()
            if unique_count > 50 and unique_count < df.shape[0]:
                report['column_flags'][col]['high_cardinality'] = True
                report['warnings'].append({
                    'level': 'WARNING', 'column': col, 'check': 'high_cardinality',
                    'message': f"Categorical column '{col}' has high cardinality ({unique_count} unique values)."
                })
                
    report['warning_count'] = len(report['warnings'])
    
    # Log warnings
    for w in report['warnings']:
        log.warning(w['message'])
        
    return report
