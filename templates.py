"""
Pre-defined pattern templates for common array configurations.

This module provides convenience functions to create common periodic array
patterns like simple contact arrays, staggered arrays, and cluster arrays.
"""

from typing import Tuple
from array_generator.core import UnitCell, Array
from array_generator.constructors import rectangle_centered


def simple_contact_array(
    cd: float,
    pitch: float,
    rows: int,
    cols: int,
    origin: Tuple[float, float] = (0.0, 0.0)
) -> Array:
    """
    Generate simple square contact array (1 contact per cell).
    
    Creates a periodic array with one square contact centered in each unit cell.
    
    Args:
        cd: Contact critical dimension (square contact, nm)
        pitch: Pitch in both X and Y directions (nm)
        rows: Number of rows
        cols: Number of columns
        origin: Array origin (default: (0, 0))
        
    Returns:
        Array object
        
    Raises:
        ValueError: If cd >= pitch or if parameters are invalid
        
    Example:
        >>> array = simple_contact_array(cd=50.0, pitch=100.0, rows=10, cols=10)
        >>> array.num_polygons()
        100
    """
    if cd >= pitch:
        raise ValueError(f"cd ({cd}) must be less than pitch ({pitch})")
    
    unit_cell = UnitCell(
        pitch_x=pitch,
        pitch_y=pitch,
        polygons=(rectangle_centered(0.0, 0.0, cd, cd),)
    )
    
    return Array(
        unit_cell=unit_cell,
        rows=rows,
        cols=cols,
        origin=origin
    )


def staggered_contact_array(
    cd: float,
    pitch_x: float,
    pitch_y: float,
    rows: int,
    cols: int,
    stagger_fraction: float = 0.25,
    origin: Tuple[float, float] = (0.0, 0.0)
) -> Array:
    """
    Generate brick/staggered contact array (2 contacts per cell).
    
    Creates a periodic array with two square contacts per unit cell, vertically
    offset to create a brick-like pattern.
    
    Args:
        cd: Contact critical dimension (nm)
        pitch_x: Horizontal pitch (nm)
        pitch_y: Vertical pitch (nm)
        rows: Number of rows
        cols: Number of columns
        stagger_fraction: Vertical offset as fraction of pitch_y (default: 0.25)
        origin: Array origin (default: (0, 0))
        
    Returns:
        Array object
        
    Raises:
        ValueError: If cd >= min(pitch_x, pitch_y) or if parameters are invalid
        
    Example:
        >>> array = staggered_contact_array(
        ...     cd=50.0, pitch_x=100.0, pitch_y=100.0, rows=10, cols=10
        ... )
        >>> array.num_polygons()
        200
    """
    if cd >= min(pitch_x, pitch_y):
        raise ValueError(
            f"cd ({cd}) must be less than min(pitch_x, pitch_y) "
            f"= {min(pitch_x, pitch_y)}"
        )
    
    offset = pitch_y * stagger_fraction
    
    unit_cell = UnitCell(
        pitch_x=pitch_x,
        pitch_y=pitch_y,
        polygons=(
            rectangle_centered(0.0, offset, cd, cd),   # upper contact
            rectangle_centered(0.0, -offset, cd, cd),  # lower contact
        )
    )
    
    return Array(
        unit_cell=unit_cell,
        rows=rows,
        cols=cols,
        origin=origin
    )


def cluster_contact_array(
    cd: float,
    spacing: float,
    pitch: float,
    rows: int,
    cols: int,
    cluster_size: int = 3,
    origin: Tuple[float, float] = (0.0, 0.0)
) -> Array:
    """
    Generate cluster contact array (N contacts per cell in a row).
    
    Creates a periodic array with multiple square contacts per unit cell,
    arranged in a horizontal row.
    
    Args:
        cd: Contact critical dimension (nm)
        spacing: Spacing between contacts in cluster (nm)
        pitch: Unit cell pitch (nm)
        rows: Number of rows
        cols: Number of columns
        cluster_size: Number of contacts per cluster (default: 3)
        origin: Array origin (default: (0, 0))
        
    Returns:
        Array object
        
    Raises:
        ValueError: If cluster doesn't fit in unit cell or if parameters are invalid
        
    Example:
        >>> array = cluster_contact_array(
        ...     cd=30.0, spacing=50.0, pitch=200.0, rows=5, cols=5, cluster_size=3
        ... )
        >>> array.num_polygons()
        75
    """
    if cluster_size < 1:
        raise ValueError(f"cluster_size must be >= 1, got {cluster_size}")
    
    # Calculate total cluster width
    cluster_width = (cluster_size - 1) * spacing + cd
    if cluster_width >= pitch:
        raise ValueError(
            f"Cluster width ({cluster_width}) must be less than pitch ({pitch})"
        )
    
    # Generate contact positions
    polygons = []
    start_x = -(cluster_size - 1) * spacing / 2
    for i in range(cluster_size):
        x = start_x + i * spacing
        polygons.append(rectangle_centered(x, 0.0, cd, cd))
    
    unit_cell = UnitCell(
        pitch_x=pitch,
        pitch_y=pitch,
        polygons=tuple(polygons)
    )
    
    return Array(
        unit_cell=unit_cell,
        rows=rows,
        cols=cols,
        origin=origin
    )
