"""
ds_engine.utils.plot_utils
==========================

Shared plotting engine for DSEngine.
Provides a single source of truth for figure creation, styling, and finalization.
Blocks must use these functions instead of calling matplotlib directly.
"""

import matplotlib.pyplot as plt
import matplotlib.figure
import seaborn as sns
from typing import Optional

STYLE           = 'seaborn-v0_8-whitegrid'
DEFAULT_COLOR   = '#2E75B6'
PALETTE         = 'muted'
FIGSIZE_SINGLE  = (10, 6)    # single plot
FIGSIZE_GRID    = (14, 10)   # multi-panel (e.g. decomposition)
FIGSIZE_WIDE    = (14, 5)    # wide single plot (e.g. correlation heatmap)
EMBED_DPI       = 150        # used by html_report.py for base64 embedding
EXPORT_DPI      = 150        # used by plot_exporter.py for file saving


def create_figure(
    title: str, 
    figsize: Optional[tuple[int, int]] = None
) -> tuple[matplotlib.figure.Figure, plt.Axes]:
    """Create a new centralized figure.
    
    Args:
        title (str): The main figure suptitle.
        figsize (tuple, optional): Dimensions of the figure. Defaults to FIGSIZE_SINGLE.
        
    Returns:
        tuple[matplotlib.figure.Figure, plt.Axes]: The generated figure and axis.
    """
    plt.style.use(STYLE)
    fig, ax = plt.subplots(figsize=figsize or FIGSIZE_SINGLE)
    fig.suptitle(title)
    return fig, ax


def create_figure_grid(
    title: str, 
    nrows: int, 
    ncols: int, 
    figsize: Optional[tuple[int, int]] = None
) -> tuple[matplotlib.figure.Figure, plt.Axes]:
    """Create a new centralized figure with multiple panels.
    
    Args:
        title (str): The main figure suptitle.
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        figsize (tuple, optional): Dimensions of the figure. Defaults to FIGSIZE_GRID.
        
    Returns:
        tuple[matplotlib.figure.Figure, plt.Axes]: The generated figure and array of axes.
    """
    plt.style.use(STYLE)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize or FIGSIZE_GRID)
    fig.suptitle(title)
    return fig, axes


def finalize_figure(fig: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
    """Apply tight layout adjustments before saving.
    
    Args:
        fig (matplotlib.figure.Figure): The figure to finalize.
        
    Returns:
        matplotlib.figure.Figure: The adjusted figure.
    """
    fig.tight_layout()
    return fig


def get_palette(n_colors: int) -> list[str]:
    """Return a list of hex colors from the standard palette.
    
    Args:
        n_colors (int): Number of colors requested.
        
    Returns:
        list[str]: Selection of hex format colors.
    """
    return sns.color_palette(PALETTE, n_colors=n_colors).as_hex()
