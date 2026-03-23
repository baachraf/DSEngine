"""
ds_engine.data.inspector
========================

Dataset pre-flight inspector. Runs automatically before any pipeline steps.
"""

import sys
import pandas as pd
import numpy as np
from typing import Any
from ds_engine.utils import logger

def inspect(
    df: pd.DataFrame, 
    columns: list[str], 
    experiment_name: str
) -> dict[str, Any]:
    """Inspect DataFrame before pipeline execution. Print summary and all warnings.
    
    Never modifies df. Returns inspection_report dict saved in report.json.
    
    Args:
        df (pd.DataFrame): Data to inspect.
        columns (list[str]): Columns the user requested (for validation).
        experiment_name (str): Current experiment name.
        
    Returns:
        dict[str, Any]: Structured inspection report.
    """
    if len(df) == 0:
        logger.log_error("[FATAL] DataFrame has 0 rows after loading.")
        sys.exit(2)
        
    if len(df.columns) == 0:
        logger.log_error("[FATAL] No columns remain to evaluate.")
        sys.exit(2)
        
    if columns:
        missing = [col for col in columns if col not in df.columns]
        if missing:
            logger.log_error(f"[FATAL] Requested columns do not exist in the DataFrame: {missing}")
            sys.exit(2)

    n_rows, n_cols = df.shape
    report = {
        'shape': {'rows': int(n_rows), 'columns': int(n_cols)},
        'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'null_summary': {},
        'warnings': [],
        'column_flags': {
            col: {
                'all_null': False,
                'zero_variance': False,
                'has_inf': False,
                'dtype_mismatch': False,
                'high_cardinality': False
            } for col in df.columns
        },
        'warning_count': 0,
        'fatal_count': 0
    }
    
    def add_warning(column: str, check: str, message: str) -> None:
        report['warnings'].append({
            'level': 'WARNING',
            'column': column,
            'check': check,
            'message': message
        })
        report['warning_count'] += 1
        logger.log_warning(f"[{check.upper()}] {message}")

    if n_rows < 30:
        add_warning('__dataset__', 'low_row_count', "Dataset has fewer than 30 rows. Statistical test reliability is severely affected.")
    if n_rows > 1_000_000:
        add_warning('__dataset__', 'high_row_count', f"Dataset has {n_rows} rows. Loading into memory may be slow or cause MemoryError. Consider using the 'sample' step as the first step.")
        
    duplicates = int(df.duplicated().sum())
    if duplicates > 0:
        dup_pct = (duplicates / n_rows) * 100
        add_warning('__dataset__', 'duplicate_rows', f"Dataset contains {duplicates} duplicate rows ({dup_pct:.1f}%).")

    for col in df.columns:
        s = df[col]
        null_count = int(s.isna().sum())
        null_pct = (null_count / n_rows) * 100
        report['null_summary'][col] = {'null_count': null_count, 'null_pct': float(null_pct)}
        
        # Check Nulls
        if null_pct == 100.0:
            report['column_flags'][col]['all_null'] = True
            add_warning(col, 'all_null', f"Column '{col}' is 100% null.")
        elif null_pct > 50.0:
            add_warning(col, 'high_null', f"Column '{col}' has {null_pct:.1f}% null values.")
            
        if null_pct == 100.0:
            continue
            
        valid_s = s.dropna()
        
        # Check Zero Variance
        if pd.api.types.is_numeric_dtype(valid_s):
            if valid_s.nunique() == 1:
                report['column_flags'][col]['zero_variance'] = True
                val = valid_s.iloc[0]
                add_warning(col, 'zero_variance', f"Column '{col}' is numeric but has zero variance. All values are {val}.")
                
            # Check Infinities
            inf_count = int(np.isinf(valid_s).sum())
            if inf_count > 0:
                report['column_flags'][col]['has_inf'] = True
                add_warning(col, 'has_inf', f"Column '{col}' contains {inf_count} infinite values.")
                
            # Check Skewness
            if len(valid_s) > 2:
                skew = valid_s.skew()
                if pd.notna(skew) and abs(skew) > 10:
                    add_warning(col, 'high_skew', f"Column '{col}' has extreme skewness ({skew:.2f}). Consider a log transform.")
        
        elif pd.api.types.is_object_dtype(valid_s):
            # Check for pseudo-numeric
            if valid_s.str.isnumeric().all():
                report['column_flags'][col]['dtype_mismatch'] = True
                add_warning(col, 'pseudo_numeric', f"Column '{col}' is stored as string but appears numeric. Suggest pd.to_numeric().")
            
            # Check for pseudo-datetime (simple heuristic)
            elif valid_s.astype(str).str.contains(r'^\d{4}-\d{2}-\d{2}', na=False).all():
                report['column_flags'][col]['dtype_mismatch'] = True
                sample_val = valid_s.iloc[0]
                add_warning(col, 'pseudo_datetime', f"Column '{col}' appears to be date strings (e.g. {sample_val}). Suggest pd.to_datetime().")
                
            # Check high cardinality
            unique_count = valid_s.nunique()
            if unique_count > 50:
                report['column_flags'][col]['high_cardinality'] = True
                add_warning(col, 'high_cardinality', f"Column '{col}' has high cardinality (> 50 unique values: {unique_count}).")
                
    return report
