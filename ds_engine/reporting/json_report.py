"""
ds_engine.reporting.json_report
===============================

Serialize all collected output_data dictionaries into a single JSON file.
Includes a recursive converter to handle numpy types safely.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Any
from ds_engine.utils import logger

def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy data types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return str(obj)
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return [_make_serializable(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    else:
        return obj

def export(
    results: dict[str, dict[str, Any]], 
    output_dir: str, 
    experiment_name: str
) -> str:
    """Save structured experiment results to JSON.
    
    Args:
        results (dict): All output_data from steps, plus pre_flight_inspection.
        output_dir (str): Absolute output directory path.
        experiment_name (str): The name of the experiment.
        
    Returns:
        str: Absolute path to the saved JSON file, or empty string on failure.
    """
    report = {
        'experiment_name': experiment_name,
        'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'steps': results
    }
    
    clean_report = _make_serializable(report)
    out_path = Path(output_dir) / 'report.json'
    
    try:
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(clean_report, f, indent=2)
        return str(out_path)
    except Exception as e:
        logger.log_error(f"Failed to write report.json: {e}")
        return ""
