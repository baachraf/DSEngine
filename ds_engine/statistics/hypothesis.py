"""
ds_engine.statistics.hypothesis
===============================

Run statistical hypothesis tests between specified column groups.
This module follows the standard block contract.

Expected params keys:
    test (str): 't-test', 'chi-square', 'anova', or 'mannwhitney' (Required).
    alpha (float): Significance level. Default 0.05.
    group_column (str): Category column for t-test/anova. Required for those.
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.figure
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from typing import Any
from ds_engine.utils import logger

def _calculate_cohens_d(x, y):
    """Calculate Cohen's d for independent samples."""
    nx, ny = len(x), len(y)
    varx, vary = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * varx + (ny - 1) * vary) / (nx + ny - 2))
    return (np.mean(x) - np.mean(y)) / pooled_std if pooled_std > 0 else 0.0

def _calculate_eta_squared(group_data):
    """Calculate Eta-squared for one-way ANOVA."""
    all_vals = np.concatenate(group_data)
    grand_mean = np.mean(all_vals)
    ss_total = np.sum((all_vals - grand_mean)**2)
    ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in group_data)
    return ss_between / ss_total if ss_total > 0 else 0.0

def run(
    df: pd.DataFrame, 
    columns: list[str], 
    params: dict[str, Any]
) -> tuple[dict[str, Any], list[matplotlib.figure.Figure]]:
    test_type = params.get('test')
    alpha = params.get('alpha', 0.05)
    group_col = params.get('group_column')
    
    if not test_type:
        raise ValueError("hypothesis step requires 'test' param.")
        
    output_data: dict[str, Any] = {}
    
    target_cols = columns if columns else df.columns
    
    if test_type in ['t-test', 'anova', 'mannwhitney']:
        if not group_col or group_col not in df.columns:
            raise KeyError(f"{test_type} requires valid 'group_column'.")
            
        groups = sorted(df[group_col].dropna().unique())
        if test_type == 't-test' and len(groups) != 2:
            logger.log_warning(f"t-test requires exactly 2 groups in '{group_col}', found {len(groups)}.")
            return {}, []
            
        num_cols = [c for c in target_cols if pd.api.types.is_numeric_dtype(df[c]) and c != group_col]
        for col in num_cols:
            valid_df = df[[col, group_col]].dropna()
            group_data = [valid_df[valid_df[group_col] == g][col] for g in groups]
            
            p_val = None
            stat = None
            effect_size = None
            effect_name = None
            post_hoc = None
            
            try:
                if test_type == 't-test':
                    stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
                    effect_size = _calculate_cohens_d(group_data[0], group_data[1])
                    effect_name = "Cohen's d"
                elif test_type == 'anova':
                    if len(groups) > 1:
                        stat, p_val = stats.f_oneway(*group_data)
                        effect_size = _calculate_eta_squared(group_data)
                        effect_name = "Eta-squared"
                        
                        # Post-hoc if significant
                        if p_val < alpha and len(groups) > 2:
                            tukey = pairwise_tukeyhsd(endog=valid_df[col], groups=valid_df[group_col], alpha=alpha)
                            # Convert Tukey summary to serializable dict
                            post_hoc = []
                            for i, row in enumerate(tukey.summary().data[1:]): # skip header
                                post_hoc.append({
                                    'group1': str(row[0]),
                                    'group2': str(row[1]),
                                    'meandiff': float(row[2]),
                                    'p_adj': float(row[3]),
                                    'lower': float(row[4]),
                                    'upper': float(row[5]),
                                    'reject': bool(row[6])
                                })
                                
                elif test_type == 'mannwhitney':
                    if len(groups) == 2:
                        stat, p_val = stats.mannwhitneyu(group_data[0], group_data[1], alternative='two-sided')
            except Exception as e:
                logger.log_warning(f"Failed {test_type} on {col}: {e}")
                
            if p_val is not None:
                reject = p_val < alpha
                interp = f"Significant difference in '{col}' across '{group_col}'" if reject else f"No significant difference"
                
                output_data[col] = {
                    'test_name': test_type,
                    'statistic': float(stat),
                    'p_value': float(p_val),
                    'alpha': float(alpha),
                    'reject_null': bool(reject),
                    'interpretation': interp
                }
                if effect_size is not None:
                    output_data[col]['effect_size'] = float(effect_size)
                    output_data[col]['effect_name'] = effect_name
                if post_hoc:
                    output_data[col]['post_hoc_tukey'] = post_hoc
                
    elif test_type == 'chi-square':
        cat_cols = [c for c in target_cols if hasattr(df[c], 'cat') or pd.api.types.is_object_dtype(df[c])]
        if len(cat_cols) < 2:
            return {}, []
            
        if group_col and group_col in cat_cols:
            other_cols = [c for c in cat_cols if c != group_col]
            for col in other_cols:
                ct = pd.crosstab(df[col], df[group_col])
                if np.min(ct.shape) > 1:
                    chi2, p, dof, _ = stats.chi2_contingency(ct)
                    reject = p < alpha
                    output_data[f"{col}_vs_{group_col}"] = {
                        'test_name': 'chi-square',
                        'statistic': float(chi2),
                        'p_value': float(p),
                        'alpha': float(alpha),
                        'reject_null': bool(reject),
                        'interpretation': f"Dependent" if reject else "Independent"
                    }
        else:
            logger.log_warning("chi-square requires a valid 'group_column'.")
            
    return output_data, []
