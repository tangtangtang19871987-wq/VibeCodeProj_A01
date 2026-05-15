"""
Visualization functions for arrays and unit cells.

This module provides functions to plot unit cells and arrays using matplotlib.
"""

from typing import Optional
import numpy as np
from array_generator.core import UnitCell, Array

try:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as MPLPolygon
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


def plot_unit_cell(
    unit_cell: UnitCell,
    show_bounds: bool = True,
    ax=None,
    title: Optional[str] = None
) -> None:
    """
    Plot unit cell with matplotlib.
    
    Args:
        unit_cell: Unit cell to plot
        show_bounds: Whether to show unit cell bounds (default: True)
        ax: Matplotlib axes (optional, creates new figure if None)
        title: Plot title (optional)
        
    Raises:
        ImportError: If matplotlib is not installed
        
    Example:
        >>> from array_generator.templates import simple_contact_array
        >>> array = simple_contact_array(cd=50, pitch=100, rows=1, cols=1)
        >>> plot_unit_cell(array.unit_cell)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    
    # Plot unit cell bounds
    if show_bounds:
        min_x, min_y, max_x, max_y = unit_cell.bounds()
        ax.plot(
            [min_x, max_x, max_x, min_x, min_x],
            [min_y, min_y, max_y, max_y, min_y],
            'k--', linewidth=1, label='Unit cell bounds'
        )
    
    # Plot polygons
    for i, poly in enumerate(unit_cell.polygons):
        vertices = poly.vertices
        patch = MPLPolygon(
            vertices,
            closed=True,
            facecolor='blue',
            edgecolor='black',
            alpha=0.6,
            linewidth=1.5
        )
        ax.add_patch(patch)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    
    if title:
        ax.set_title(title)
    else:
        ax.set_title(
            f'Unit Cell: {unit_cell.pitch_x}×{unit_cell.pitch_y} nm, '
            f'{unit_cell.num_polygons()} polygon(s)'
        )
    
    if show_bounds:
        ax.legend()
    
    plt.tight_layout()


def plot_array(
    array: Array,
    max_cells: int = 100,
    show_grid: bool = True,
    ax=None,
    title: Optional[str] = None
) -> None:
    """
    Plot array (or subset if too large).
    
    If the array has more than max_cells, only a subset is plotted.
    
    Args:
        array: Array to plot
        max_cells: Maximum number of cells to plot (default: 100)
        show_grid: Whether to show unit cell grid (default: True)
        ax: Matplotlib axes (optional, creates new figure if None)
        title: Plot title (optional)
        
    Raises:
        ImportError: If matplotlib is not installed
        
    Example:
        >>> from array_generator.templates import simple_contact_array
        >>> array = simple_contact_array(cd=50, pitch=100, rows=10, cols=10)
        >>> plot_array(array)
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    
    # Determine how many cells to plot
    total_cells = array.num_cells()
    if total_cells > max_cells:
        # Plot a subset - calculate rows and cols to plot
        aspect_ratio = array.cols / array.rows
        plot_rows = int(np.sqrt(max_cells / aspect_ratio))
        plot_cols = int(plot_rows * aspect_ratio)
        plot_rows = max(1, plot_rows)
        plot_cols = max(1, plot_cols)
        
        # Center the subset
        start_row = (array.rows - plot_rows) // 2
        start_col = (array.cols - plot_cols) // 2
        
        warning_text = f"Showing {plot_rows}×{plot_cols} of {array.rows}×{array.cols} cells"
    else:
        plot_rows = array.rows
        plot_cols = array.cols
        start_row = 0
        start_col = 0
        warning_text = None
    
    # Generate polygons for the subset
    for j in range(start_row, start_row + plot_rows):
        for i in range(start_col, start_col + plot_cols):
            if j >= array.rows or i >= array.cols:
                continue
                
            cell_center = array.get_cell_center(i, j)
            
            # Plot each polygon in the cell
            for poly in array.unit_cell.polygons:
                world_poly = poly.translate(cell_center)
                vertices = world_poly.vertices
                
                patch = MPLPolygon(
                    vertices,
                    closed=True,
                    facecolor='blue',
                    edgecolor='black',
                    alpha=0.6,
                    linewidth=0.5
                )
                ax.add_patch(patch)
    
    # Show unit cell grid
    if show_grid:
        pitch_x = array.unit_cell.pitch_x
        pitch_y = array.unit_cell.pitch_y
        
        # Vertical lines
        for i in range(start_col, start_col + plot_cols + 1):
            if i > array.cols:
                break
            x = array.origin[0] + i * pitch_x
            y_start = array.origin[1] + start_row * pitch_y
            y_end = array.origin[1] + min(start_row + plot_rows, array.rows) * pitch_y
            ax.plot([x, x], [y_start, y_end], 'k-', linewidth=0.5, alpha=0.3)
        
        # Horizontal lines
        for j in range(start_row, start_row + plot_rows + 1):
            if j > array.rows:
                break
            y = array.origin[1] + j * pitch_y
            x_start = array.origin[0] + start_col * pitch_x
            x_end = array.origin[0] + min(start_col + plot_cols, array.cols) * pitch_x
            ax.plot([x_start, x_end], [y, y], 'k-', linewidth=0.5, alpha=0.3)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Labels
    ax.set_xlabel('X (nm)')
    ax.set_ylabel('Y (nm)')
    
    if title:
        ax.set_title(title)
    else:
        title_text = f'Array: {array.rows}×{array.cols} cells, {array.num_polygons()} polygons'
        if warning_text:
            title_text += f'\n({warning_text})'
        ax.set_title(title_text)
    
    plt.tight_layout()


def save_unit_cell_plot(
    unit_cell: UnitCell,
    filename: str,
    show_bounds: bool = True,
    dpi: int = 150
) -> None:
    """
    Save unit cell plot to file.
    
    Args:
        unit_cell: Unit cell to plot
        filename: Output filename (e.g., 'unit_cell.png')
        show_bounds: Whether to show unit cell bounds
        dpi: Resolution in dots per inch (default: 150)
        
    Raises:
        ImportError: If matplotlib is not installed
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )
    
    fig, ax = plt.subplots(figsize=(8, 8))
    plot_unit_cell(unit_cell, show_bounds=show_bounds, ax=ax)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def save_array_plot(
    array: Array,
    filename: str,
    max_cells: int = 100,
    show_grid: bool = True,
    dpi: int = 150
) -> None:
    """
    Save array plot to file.
    
    Args:
        array: Array to plot
        filename: Output filename (e.g., 'array.png')
        max_cells: Maximum number of cells to plot
        show_grid: Whether to show unit cell grid
        dpi: Resolution in dots per inch (default: 150)
        
    Raises:
        ImportError: If matplotlib is not installed
    """
    if not MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )
    
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_array(array, max_cells=max_cells, show_grid=show_grid, ax=ax)
    plt.savefig(filename, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
