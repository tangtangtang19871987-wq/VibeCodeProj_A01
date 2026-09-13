"""
Convenience constructors for common polygon shapes.

This module provides functions to easily create common polygon shapes like
rectangles without manually specifying vertices.
"""

import numpy as np
from array_generator.core import Polygon


def rectangle_centered(
    center_x: float,
    center_y: float,
    width: float,
    height: float
) -> Polygon:
    """
    Create axis-aligned rectangle centered at (center_x, center_y).
    
    Args:
        center_x: X coordinate of center (nm)
        center_y: Y coordinate of center (nm)
        width: Rectangle width (nm)
        height: Rectangle height (nm)
        
    Returns:
        Polygon in canonical form
        
    Raises:
        ValueError: If width or height is not positive
        
    Example:
        >>> rect = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        >>> rect.bbox()
        (-50.0, -25.0, 50.0, 25.0)
    """
    if width <= 0:
        raise ValueError(f"width must be positive, got {width}")
    if height <= 0:
        raise ValueError(f"height must be positive, got {height}")
    
    hw = width / 2
    hh = height / 2
    
    vertices = np.array([
        [center_x - hw, center_y - hh],  # bottom-left
        [center_x + hw, center_y - hh],  # bottom-right
        [center_x + hw, center_y + hh],  # top-right
        [center_x - hw, center_y + hh],  # top-left
    ], dtype=np.float64)
    
    return Polygon(vertices=vertices)


def rectangle_corner(
    x: float,
    y: float,
    width: float,
    height: float
) -> Polygon:
    """
    Create axis-aligned rectangle with bottom-left corner at (x, y).
    
    Args:
        x: X coordinate of bottom-left corner (nm)
        y: Y coordinate of bottom-left corner (nm)
        width: Rectangle width (nm)
        height: Rectangle height (nm)
        
    Returns:
        Polygon in canonical form
        
    Raises:
        ValueError: If width or height is not positive
        
    Example:
        >>> rect = rectangle_corner(0.0, 0.0, 100.0, 50.0)
        >>> rect.bbox()
        (0.0, 0.0, 100.0, 50.0)
    """
    return rectangle_centered(
        center_x=x + width / 2,
        center_y=y + height / 2,
        width=width,
        height=height
    )
