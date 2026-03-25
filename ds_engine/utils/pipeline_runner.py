"""
ds_engine.utils.pipeline_runner
===============================

Orchestrate a full pipeline run from parsed config to final report.
This is the core engine of the library. It is the only function called
by run_pipeline.py and integration tests.
"""

import sys
import time
import importlib
import traceback
from datetime import datetime
from pathlib import Path

from ds_engine.utils import config_parser, logger
from ds_engine.data import loader, inspector
from ds_engine.reporting import html_report, json_report, plot_exporter
import os
import matplotlib.pyplot as plt

# Suggestion map for common block exceptions
_EXCEPTION_SUGGESTIONS = {
    'KeyError': 'Check that the column name exists in the dataset and is spelled correctly in your config.',
    'ValueError': 'Check the params for this step in pipeline.yml — a value may be invalid or out of range.',
    'TypeError': 'A column may have an incompatible dtype (e.g. string column passed to a numeric-only block).',
    'MemoryError': 'Dataset may be too large. Add a sample step before this step in your config.'
}

def _auto_sanitize_params(params: dict, df_columns: list[str]) -> dict:
    """Recursively sanitize any string values in params that match unnormalized column names.
    This helps when users put 'Date' in params but the loader changed it to 'date'.
    """
    if not isinstance(params, dict):
        return params
        
    sanitized = {}
    for k, v in params.items():
        if isinstance(v, str):
            # Try to sanitize the value and see if it exists in the clean columns
            v_sanitized = loader.sanitize_column_name(v)
            if v_sanitized in df_columns:
                sanitized[k] = v_sanitized
            else:
                sanitized[k] = v
        elif isinstance(v, list):
            # Handle lists of potentially unnormalized column names
            new_list = []
            for item in v:
                if isinstance(item, str):
                    item_sanitized = loader.sanitize_column_name(item)
                    if item_sanitized in df_columns:
                        new_list.append(item_sanitized)
                    else:
                        new_list.append(item)
                else:
                    new_list.append(item)
            sanitized[k] = new_list
        elif isinstance(v, dict):
            sanitized[k] = _auto_sanitize_params(v, df_columns)
        else:
            sanitized[k] = v
    return sanitized

def _is_notebook() -> bool:
    """Check if we are running in a Jupyter notebook or IPython environment."""
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if shell in ['ZMQInteractiveShell', 'TerminalInteractiveShell']:
            return True
        return False
    except Exception:
        return False

def run(config_path: str, experiment_name: str, display_plots: bool = True) -> int:
    """Run a DSEngine pipeline experiment.
    Loads config, runs all steps, saves all outputs, prints console summary.
    
    Args:
        config_path (str): Absolute or relative path to the pipeline.yml file.
        experiment_name (str): Name of the experiment to run.
        
    Returns:
        int: Exit code. 0 = success, 1 = completed with failures, 2 = aborted.
        
    Raises:
        SystemExit: On fatal errors (exit code 2). Never raises on step failures.
    """
    start_time = datetime.now()
    t_start = time.time()
    
    # 1. Print Header
    print(f"+{'-'*62}+")
    print(f"|  DSEngine  |  Experiment: {experiment_name:<29} |")
    print(f"|  Config  : {config_path:<47} |")
    print(f"|  Started : {start_time.strftime('%Y-%m-%d %H:%M:%S'):<47} |")
    print(f"+{'-'*62}+")
    
    # 2. Parse Config
    try:
        config = config_parser.parse(config_path, experiment_name)
    except Exception as e:
        logger.log_error(f"[FATAL] Config matching error: {e}")
        print(f"+{'-'*62}+")
        print(f"|  x  Run aborted — fatal issue detected                       |")
        print(f"+{'-'*62}+")
        sys.exit(2)
        
    # 3. Load Data
    print(f"> Loading data...")
    try:
        # We pass original columns to loader.load, which now handles sanitization internally.
        df = loader.load(config['data']['source'], config['data']['columns'], config['data']['loader_params'])
        # Update config with sanitized column list so downstream modules see consistent names
        config['data']['columns'] = loader.sanitize_column_list(config['data']['columns'])
    except Exception as e:
        logger.log_error(f"[FATAL] Loader failed: {e}")
        print(f"+{'-'*62}+")
        print(f"|  x  Run aborted — fatal issue detected                       |")
        print(f"+{'-'*62}+")
        sys.exit(2)
        
    print(f"  Source  : {config['data']['source']}")
    print(f"  Shape   : {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"  Columns : {', '.join(df.columns)}")
    
    # 4. Pre-flight inspection
    print(f"> Pre-flight inspection...")
    try:
        inspection_report = inspector.inspect(df, config['data']['columns'], experiment_name)
    except Exception as e:
        # Inspector naturally exits with 2 on FATAL. This catch is for unexpected errors.
        logger.log_error(f"[FATAL] Inspector failed: {e}")
        sys.exit(2)
        
    if inspection_report['warning_count'] == 0:
        print("  v  No fatal issues.")
    else:
        print(f"  v  No fatal issues.")
        print(f"  !  {inspection_report['warning_count']} warning(s) — check before interpreting results:")
        for idx, w in enumerate(inspection_report['warnings'], 1):
            print(f"     [{idx}] {w['message']}")
            
    # 5. Output directory setup
    timestamp_str = start_time.strftime('%Y%m%d_%H%M%S')
    out_dir = Path(config['output']['path']) / experiment_name / timestamp_str
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.log_error(f"[FATAL] Output directory cannot be created: {e}")
        sys.exit(2)
        
    logger.set_output_dir(str(out_dir))
    
    # 6. Step execution
    steps = config['steps']
    num_steps = len(steps)
    print(f"> Running steps ({num_steps} total):")
    
    failed_steps = []
    results = {'pre_flight_inspection': inspection_report}
    collected_figures = {}
    
    for idx, step in enumerate(steps, 1):
        step_name = step['name']
        step_type = step['type']
        
        print(f"  [{idx}/{num_steps}] {step_name:<16} ({step_type:<14}) ... ", end="", flush=True)
        logger.log_step_start(step_name)
        
        try:
            mod_path = config_parser.STEP_REGISTRY[step_type]
            mod = importlib.import_module(mod_path)
            
            t0 = time.time()
            # Sanitize step columns to match normalized DataFrame
            step_cols = loader.sanitize_column_list(step['columns']) if step['columns'] else []
            # Fuzzy sanitize params (e.g. mapping 'Date' to 'date' if it matches a column)
            step_params = _auto_sanitize_params(step['params'], list(df.columns))
            
            output = mod.run(df, step_cols, step_params)
            duration = time.time() - t0
            
            if len(output) == 3:
                output_data, output_plots, df = output
            else:
                output_data, output_plots = output
                
            results[step_name] = output_data
            if output_plots:
                collected_figures[step_name] = output_plots
            
            logger.log_step_end(step_name, duration)
            print(f"done  ({duration:.2f}s)")
            
        except Exception as e:
            err_type = type(e).__name__
            suggestion = _EXCEPTION_SUGGESTIONS.get(err_type, 'An unexpected error occurred. Check run.log for the full traceback.')
            
            print("FAILED")
            print(f"        Error type : {err_type}")
            print(f"        Message    : {str(e)}")
            print(f"        Suggestion : {suggestion}")
            print(f"        Continuing to next step.")
            
            logger.log_error(f"Step '{step_name}' failed: {str(e)}\n{traceback.format_exc()}")
            results[step_name] = {'status': 'FAILED', 'error': str(e)}
            failed_steps.append(step_name)
            
    # 7. Saving Outputs
    print(f"> Saving outputs...")
    
    # Save plots if needed
    plot_paths_map = {}
    total_plots = sum(len(figs) for figs in collected_figures.values())
    if config['output'].get('save_plots', True) and total_plots > 0:
        plot_format = config['output'].get('plot_format', 'png')
        plot_paths_map = plot_exporter.export(collected_figures, str(out_dir), plot_format)
        plots_folder = out_dir / 'plots'
        print(f"  Plots : {total_plots} figure(s) -> {plots_folder}/")
    elif total_plots > 0:
        # Not saving to disk, but keep them in memory for HTML render
        pass
        
    json_path = json_report.export(results, str(out_dir), experiment_name)
    print(f"  JSON  : {json_path}")
    
    if 'html' in config['output']['format']:
        # We pass figures to html_report to embed directly.
        html_path = html_report.export(results, collected_figures, str(out_dir), experiment_name, plot_paths_map)
        print(f"  HTML  : {html_path}")
        
    # Manually close all plotting figures to prevent memory leaks
    is_nb = _is_notebook()
    for figs in collected_figures.values():
        for fig in figs:
            if display_plots and is_nb:
                try:
                    from IPython.display import display
                    display(fig)
                except Exception:
                    pass
            plt.close(fig)

    total_time = time.time() - t_start
    
    # 8. Summary Output
    print(f"+{'-'*62}+")
    if failed_steps:
        print(f"|  x  Run completed with errors                                |")
        print(f"|  Steps  : {num_steps - len(failed_steps)} completed, {len(failed_steps)} failed                              |")
        rel_out_dir = os.path.relpath(out_dir) if os.path.isabs(out_dir) else str(out_dir)
        print(f"|  Failed : {', '.join(failed_steps):<46} |")
        print(f"|  Output : {rel_out_dir:<50} |")
        print(f"+{'-'*62}+")
        return 1
    else:
        print(f"|  v  Run complete                                             |")
        print(f"|  Steps  : {num_steps} completed, 0 failed                             |")
        rel_out_dir = os.path.relpath(out_dir) if os.path.isabs(out_dir) else str(out_dir)
        print(f"|  Time   : {total_time:.2f}s total                                        |")
        print(f"|  Output : {rel_out_dir:<50} |")
        print(f"+{'-'*62}+")
        return 0
