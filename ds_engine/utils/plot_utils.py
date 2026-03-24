"""
ds_engine.utils.plot_utils
==========================
Shared plotting engine for DSEngine.
Every block that generates plots must use these functions instead of calling matplotlib directly.
"""

import matplotlib.pyplot as plt
from typing import Any

# Constants as defined in Section 10b.2
STYLE           = 'seaborn-v0_8-whitegrid'
DEFAULT_COLOR   = '#2E75B6'
PALETTE         = 'muted'
FIGSIZE_SINGLE  = (10, 6)
FIGSIZE_GRID    = (14, 10)
FIGSIZE_WIDE    = (14, 5)
EMBED_DPI       = 150
EXPORT_DPI      = 150

def create_figure(title: str, figsize: tuple[int, int] | None = None) -> tuple[plt.Figure, plt.Axes]:
    """Create a new figure with standard style applied.
    
    Args:
        title (str): Suptitle of the figure.
        figsize (tuple[int, int] | None): Figure size. Defaults to FIGSIZE_SINGLE.
        
    Returns:
        tuple[plt.Figure, plt.Axes]: Figure and Axes objects.
    """
    try:
        plt.style.use(STYLE)
    except OSError:
        pass # fallback if style not available
        
    if figsize is None:
        figsize = FIGSIZE_SINGLE
        
    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(title)
    return fig, ax

def create_figure_grid(title: str, nrows: int, ncols: int, figsize: tuple[int, int] | None = None) -> tuple[plt.Figure, Any]:
    """Create a multi-panel figure.
    
    Args:
        title (str): Suptitle of the figure.
        nrows (int): Number of rows.
        ncols (int): Number of columns.
        figsize (tuple[int, int] | None): Figure size. Defaults to FIGSIZE_GRID if grid.
        
    Returns:
        tuple[plt.Figure, Any]: Figure and array of Axes objects.
    """
    try:
        plt.style.use(STYLE)
    except OSError:
        pass
        
    if figsize is None:
        figsize = FIGSIZE_GRID
        
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.suptitle(title)
    return fig, axes

def finalize_figure(fig: plt.Figure) -> plt.Figure:
    """Finalize figure layout before saving or returning.
    
    Args:
        fig (plt.Figure): The Matplotlib Figure object.
        
    Returns:
        plt.Figure: The same Figure object.
    """
    fig.tight_layout()
    return fig

def get_palette(n_colors: int) -> list[str]:
    """Get a list of hex color strings from the standard palette.
    
    Args:
        n_colors (int): Number of colors requested.
        
    Returns:
        list[str]: List of hex color strings.
    """
    import seaborn as sns
    return sns.color_palette(PALETTE, n_colors).as_hex()
