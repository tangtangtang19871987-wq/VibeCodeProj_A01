"""
Export functions for arrays and polygons.

This module provides functions to export arrays to various formats including
text files (CSV, space-separated) and GDS format.
"""

from typing import List
import numpy as np
from array_generator.core import Polygon, Array


def export_polygon_vertices(
    polygons: List[Polygon],
    filename: str,
    format: str = 'csv'
) -> None:
    """
    Export polygon vertices to text file.
    
    Each polygon is written as a separate section with all its vertices.
    
    Args:
        polygons: List of polygons to export
        filename: Output filename
        format: 'csv' (comma-separated) or 'space' (space-separated)
        
    Raises:
        ValueError: If format is not 'csv' or 'space'
        IOError: If file cannot be written
        
    Example:
        >>> polygons = [rectangle_centered(0, 0, 100, 50)]
        >>> export_polygon_vertices(polygons, 'vertices.csv', format='csv')
    """
    if format not in ['csv', 'space']:
        raise ValueError(f"format must be 'csv' or 'space', got '{format}'")
    
    separator = ',' if format == 'csv' else ' '
    
    try:
        with open(filename, 'w') as f:
            # Write header
            f.write(f"# Polygon vertices export\n")
            f.write(f"# Format: polygon_id{separator}vertex_id{separator}x{separator}y\n")
            
            for poly_id, poly in enumerate(polygons):
                for vert_id, vertex in enumerate(poly.vertices):
                    f.write(f"{poly_id}{separator}{vert_id}{separator}")
                    f.write(f"{vertex[0]}{separator}{vertex[1]}\n")
    except IOError as e:
        raise IOError(f"Failed to write to {filename}: {e}")


def export_polygon_centroids(
    polygons: List[Polygon],
    filename: str,
    format: str = 'csv'
) -> None:
    """
    Export polygon centroids to text file.
    
    Args:
        polygons: List of polygons to export
        filename: Output filename
        format: 'csv' (comma-separated) or 'space' (space-separated)
        
    Raises:
        ValueError: If format is not 'csv' or 'space'
        IOError: If file cannot be written
        
    Example:
        >>> polygons = [rectangle_centered(0, 0, 100, 50)]
        >>> export_polygon_centroids(polygons, 'centroids.csv')
    """
    if format not in ['csv', 'space']:
        raise ValueError(f"format must be 'csv' or 'space', got '{format}'")
    
    separator = ',' if format == 'csv' else ' '
    
    try:
        with open(filename, 'w') as f:
            # Write header
            f.write(f"# Polygon centroids export\n")
            f.write(f"# Format: polygon_id{separator}cx{separator}cy\n")
            
            for poly_id, poly in enumerate(polygons):
                cx, cy = poly.centroid()
                f.write(f"{poly_id}{separator}{cx}{separator}{cy}\n")
    except IOError as e:
        raise IOError(f"Failed to write to {filename}: {e}")


def export_to_gds(
    array: Array,
    filename: str,
    layer: int = 1,
    datatype: int = 0,
    unit: float = 1e-9,  # 1nm
    precision: float = 1e-12
) -> None:
    """
    Export array to GDS format.
    
    Uses gdspy library to create GDS file. All coordinates are assumed to be
    in nanometers and are converted to GDS database units.
    
    Args:
        array: Array to export
        filename: Output GDS filename (should end with .gds)
        layer: GDS layer number (default: 1)
        datatype: GDS datatype (default: 0)
        unit: User unit in meters (default: 1nm = 1e-9m)
        precision: Precision in meters (default: 1pm = 1e-12m)
        
    Raises:
        ImportError: If gdspy is not installed
        IOError: If file cannot be written
        
    Example:
        >>> array = simple_contact_array(cd=50, pitch=100, rows=10, cols=10)
        >>> export_to_gds(array, 'array.gds', layer=1)
    """
    try:
        import gdspy
    except ImportError:
        raise ImportError(
            "gdspy is required for GDS export. "
            "Install it with: pip install gdspy"
        )
    
    try:
        # Create GDS library (not using current_library to avoid conflicts)
        lib = gdspy.GdsLibrary(unit=unit, precision=precision)
        
        # Create cell directly without adding to current_library
        cell = gdspy.Cell('ARRAY', exclude_from_current=True)
        lib.add(cell)
        
        # Generate all polygons
        polygons = array.generate_polygons()
        
        # Add each polygon to the cell
        for poly in polygons:
            # Convert vertices to list of tuples (required by gdspy)
            points = [(v[0], v[1]) for v in poly.vertices]
            
            # Create polygon and add to cell
            gds_poly = gdspy.Polygon(points, layer=layer, datatype=datatype)
            cell.add(gds_poly)
        
        # Write to file
        lib.write_gds(filename)
        
    except Exception as e:
        raise IOError(f"Failed to write GDS file {filename}: {e}")


def export_array_metadata(
    array: Array,
    filename: str
) -> None:
    """
    Export array metadata to text file.
    
    Includes information about unit cell, array dimensions, and bounds.
    
    Args:
        array: Array to export metadata for
        filename: Output filename
        
    Raises:
        IOError: If file cannot be written
        
    Example:
        >>> array = simple_contact_array(cd=50, pitch=100, rows=10, cols=10)
        >>> export_array_metadata(array, 'metadata.txt')
    """
    try:
        with open(filename, 'w') as f:
            f.write("# Array Metadata\n\n")
            
            # Unit cell info
            f.write("## Unit Cell\n")
            f.write(f"pitch_x: {array.unit_cell.pitch_x} nm\n")
            f.write(f"pitch_y: {array.unit_cell.pitch_y} nm\n")
            f.write(f"polygons_per_cell: {array.unit_cell.num_polygons()}\n")
            f.write(f"total_area: {array.unit_cell.total_area()} nm²\n")
            f.write(f"density: {array.unit_cell.density():.4f}\n\n")
            
            # Array info
            f.write("## Array\n")
            f.write(f"rows: {array.rows}\n")
            f.write(f"cols: {array.cols}\n")
            f.write(f"total_cells: {array.num_cells()}\n")
            f.write(f"total_polygons: {array.num_polygons()}\n")
            f.write(f"origin: ({array.origin[0]}, {array.origin[1]}) nm\n\n")
            
            # Bounds
            min_x, min_y, max_x, max_y = array.bounds()
            f.write("## Bounds\n")
            f.write(f"min_x: {min_x} nm\n")
            f.write(f"min_y: {min_y} nm\n")
            f.write(f"max_x: {max_x} nm\n")
            f.write(f"max_y: {max_y} nm\n")
            f.write(f"width: {max_x - min_x} nm\n")
            f.write(f"height: {max_y - min_y} nm\n")
            
    except IOError as e:
        raise IOError(f"Failed to write to {filename}: {e}")
