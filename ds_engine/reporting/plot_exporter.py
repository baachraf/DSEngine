"""
ds_engine.reporting.plot_exporter
=================================

Save all Figure objects collected during a pipeline run to disk.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.figure
from typing import Any
from ds_engine.utils import plot_utils, logger

def export(
    plots: dict[str, list[matplotlib.figure.Figure]], 
    output_dir: str, 
    format_type: str = 'png'
) -> dict[str, list[str]]:
    """Save all plots to disk.
    
    Args:
        plots (dict): Dict keyed by step_name containing lists of Figures.
        output_dir (str): Base output directory for this experiment run.
        format_type (str): 'png', 'svg', or 'both'. Defaults to 'png'.
        
    Returns:
        dict[str, list[str]]: Map of step_name to saved file paths.
    """
    plot_dir = Path(output_dir) / 'plots'
    if not plot_dir.exists():
        plot_dir.mkdir(parents=True)
        
    formats = ['png', 'svg'] if format_type == 'both' else [format_type]
    dpi = plot_utils.EXPORT_DPI
    
    saved_paths = {}
    
    for step_name, figs in plots.items():
        step_paths = []
        for i, fig in enumerate(figs):
            for fmt in formats:
                filepath = plot_dir / f"{step_name}_{i}.{fmt}"
                try:
                    fig.savefig(filepath, format=fmt, dpi=dpi, bbox_inches='tight')
                    step_paths.append(str(filepath))
                except Exception as e:
                    logger.log_error(f"Failed to save plot {filepath.name}: {e}")
        saved_paths[step_name] = step_paths
        
    return saved_paths
