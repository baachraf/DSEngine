"""
ds_engine.utils.config_parser
=============================

Read a YAML pipeline config file and return a validated, normalized Python dictionary.
Must apply step defaults from configs/steps_defaults.yml and resolve all paths
(data.source, output.path, validate.schema_path) relative to the project root.
"""

import yaml
from pathlib import Path
from typing import Any

# Import logger to use for fatal error messages (though parser raises ValueError)
# The pipeline_runner calls sys.exit(2) on parser exception, but we raise it here.

STEP_REGISTRY = {
    'summary'        : 'ds_engine.exploration.summary',
    'distributions'  : 'ds_engine.exploration.distributions',
    'correlations'   : 'ds_engine.exploration.correlations',
    'missing'        : 'ds_engine.exploration.missing',
    'outliers'       : 'ds_engine.exploration.outliers',
    'categorical'    : 'ds_engine.exploration.categorical',
    'hypothesis'     : 'ds_engine.statistics.hypothesis',
    'normality'      : 'ds_engine.statistics.normality',
    'relationships'  : 'ds_engine.statistics.relationships',
    'decomposition'  : 'ds_engine.time_series.decomposition',
    'stationarity'   : 'ds_engine.time_series.stationarity',
    'ts_features'    : 'ds_engine.time_series.features',
    'validate'       : 'ds_engine.data.validator',
    'sample'         : 'ds_engine.data.sampler',
    'pps'            : 'ds_engine.exploration.pps',
    'multivariate_outliers': 'ds_engine.exploration.multivariate_outliers',
    'manifold'       : 'ds_engine.exploration.manifold',
    'leakage'        : 'ds_engine.exploration.leakage',
    'drift'          : 'ds_engine.statistics.drift',
}

def parse(config_path: str, experiment_name: str) -> dict[str, Any]:
    """Parse and validate pipeline.yml, merging module defaults and resolving paths.
    
    Args:
        config_path (str): Path to the user's YAML config file.
        experiment_name (str): Key of the experiment to run.
        
    Returns:
        dict: Normalized experiment configuration dictionary.
        
    Raises:
        ValueError: On misconfiguration, missing files, unknown steps, missing params.
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise ValueError(f"Config file path does not exist: {config_path}")
        
    with open(config_file, 'r', encoding='utf-8') as f:
        full_config = yaml.safe_load(f)
        
    if experiment_name not in full_config:
        raise ValueError(f"Experiment name '{experiment_name}' not found in {config_path}")
        
    experiment = full_config[experiment_name]
    project_root = config_file.parent.parent.resolve()
    
    # Validate data block
    if 'data' not in experiment or 'source' not in experiment['data']:
        raise ValueError(f"Missing 'data.source' block in experiment '{experiment_name}'")
        
    raw_source = experiment['data']['source']
    source_path = Path(raw_source)
    if not source_path.is_absolute():
        source_path = (project_root / raw_source).resolve()
        
    if not source_path.exists():
        raise ValueError(
            f"data.source file not found: '{source_path}'\n"
            f"Resolved from: '{raw_source}' relative to project root '{project_root}'"
        )
    experiment['data']['source'] = str(source_path)
    # Default columns missing handling
    experiment['data'].setdefault('columns', [])
    experiment['data'].setdefault('loader_params', {})
    experiment['data'].setdefault('target', None)
    
    # Validate output block
    if 'output' not in experiment or 'path' not in experiment['output']:
        raise ValueError(f"Missing 'output.path' block in experiment '{experiment_name}'")
        
    out_format = experiment['output'].get('format', ['json'])
    if not out_format or not any(fmt in out_format for fmt in ['html', 'json']):
        raise ValueError(f"output.format empty or missing required format (html, json). Got: {out_format}")
        
    raw_output_path = experiment['output']['path']
    out_base = Path(raw_output_path)
    if not out_base.is_absolute():
        out_base = (project_root / raw_output_path).resolve()
    experiment['output']['path'] = str(out_base)
    experiment['output'].setdefault('save_plots', True)
    experiment['output'].setdefault('plot_format', 'png')
    
    # Load defaults
    defaults_path = config_file.parent / 'steps_defaults.yml'
    step_defaults = {}
    if defaults_path.is_file():
        with open(defaults_path, 'r', encoding='utf-8') as f:
            step_defaults = yaml.safe_load(f) or {}

    # Validate Steps
    if 'steps' not in experiment or not experiment['steps']:
        raise ValueError(f"No steps defined in experiment '{experiment_name}'. Must have at least one.")

    normalized_steps = []
    for step in experiment['steps']:
        if 'name' not in step or 'type' not in step:
            raise ValueError(f"A step is missing required keys 'name' or 'type'. Step dump: {step}")
            
        step_type = step['type']
        if step_type not in STEP_REGISTRY:
            raise ValueError(f"Unknown step type '{step_type}' not in STEP_REGISTRY.")
            
        step_params = step.get('params', {})
        
        # Required parameter checks explicitly from Section 6.1
        if step_type == 'hypothesis' and 'test' not in step_params:
            raise ValueError(f"Step '{step['name']}' of type 'hypothesis' missing required param 'test'. "
                             f"Valid values: t-test, chi-square, anova, mannwhitney.")
        if step_type == 'decomposition' and 'date_column' not in step_params:
            raise ValueError(f"Step '{step['name']}' of type 'decomposition' missing required param 'date_column'.")
        if step_type == 'stationarity' and 'date_column' not in step_params:
            raise ValueError(f"Step '{step['name']}' of type 'stationarity' missing required param 'date_column'.")
        if step_type == 'ts_features' and 'date_column' not in step_params:
            raise ValueError(f"Step '{step['name']}' of type 'ts_features' missing required param 'date_column'.")
        
        # Default Injection
        base_params = step_defaults.get(step_type, {}).copy()
        base_params.update(step_params)
        
        # Path resolution for validator using schema_path
        if step_type == 'validate':
            raw_schema = base_params.get('schema_path')
            if not raw_schema:
                raise ValueError(f"Step '{step['name']}' of type 'validate' requires 'schema_path'.")
            schema_path = Path(raw_schema)
            if not schema_path.is_absolute():
                schema_path = (project_root / raw_schema).resolve()
            if not schema_path.exists():
                raise ValueError(
                    f"schema_path file not found: '{schema_path}'\n"
                    f"Resolved from: '{raw_schema}' relative to project root '{project_root}'"
                )
            base_params['schema_path'] = str(schema_path)

        if step_type == 'drift':
            raw_ref = base_params.get('reference_source')
            if not raw_ref:
                raise ValueError(f"Step '{step['name']}' of type 'drift' requires 'reference_source'.")
            ref_path = Path(raw_ref)
            if not ref_path.is_absolute():
                ref_path = (project_root / raw_ref).resolve()
            if not ref_path.exists():
                raise ValueError(
                    f"reference_source file not found: '{ref_path}'\n"
                    f"Resolved from: '{raw_ref}' relative to project root '{project_root}'"
                )
            base_params['reference_source'] = str(ref_path)

        normalized_steps.append({
            'name': step['name'],
            'type': step_type,
            'columns': step.get('columns', []),
            'params': base_params
        })
        
    experiment['steps'] = normalized_steps
    experiment['experiment_name'] = experiment_name
    return experiment
