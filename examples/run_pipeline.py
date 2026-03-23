"""
examples/run_pipeline.py
========================

CLI entry point for DSEngine.
"""

import argparse
import sys
import yaml
from pathlib import Path

# Add project root to sys.path so 'ds_engine' can be imported without installing
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from ds_engine.utils import pipeline_runner

def get_available_configs(configs_dir: Path) -> list[str]:
    """List all .yml files in the configs directory."""
    if not configs_dir.is_dir():
        return []
    return sorted([p.name for p in configs_dir.glob("*.yml") if p.name != 'steps_defaults.yml'])

def get_available_experiments(config_path: Path) -> list[str]:
    """Extract top-level keys from a YAML config file."""
    if not config_path.is_file():
        return []
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            full_config = yaml.safe_load(f)
        return sorted(list(full_config.keys())) if isinstance(full_config, dict) else []
    except Exception:
        return []

def main():
    configs_dir = project_root / 'configs'
    available_configs = get_available_configs(configs_dir)
    
    # Default config path
    default_config_path = 'configs/pipeline.yml'
    full_default_path = project_root / default_config_path
    
    # Get experiments from default config for initial help
    available_experiments = get_available_experiments(full_default_path)

    parser = argparse.ArgumentParser(
        description="Run a DSEngine pipeline experiment.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    config_help = "Path to the configuration file.\nAvailable in configs/:\n  - " + "\n  - ".join(available_configs)
    parser.add_argument(
        '--config', 
        type=str, 
        default=default_config_path,
        help=config_help
    )
    
    exp_help = "Name of the experiment to run.\n"
    if available_experiments:
        exp_help += f"Available in {default_config_path}:\n  - " + "\n  - ".join(available_experiments)
    else:
        exp_help += "Note: Valid experiments depend on the selected --config file."
    
    parser.add_argument(
        '--experiment', 
        type=str, 
        required=True,
        help=exp_help
    )
    
    # If no arguments are provided, show help and exit
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
        
    args = parser.parse_args()
    
    # Run the pipeline - runner handles all logic and sys.exit(2) on fatal errors
    exit_code = pipeline_runner.run(config_path=args.config, experiment_name=args.experiment)
    
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
