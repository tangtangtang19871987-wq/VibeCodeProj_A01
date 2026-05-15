"""
Example: Custom Unit Cell

This example demonstrates how to create a custom unit cell with
arbitrary polygon shapes and positions.
"""

import numpy as np
from array_generator import (
    Polygon,
    UnitCell,
    Array,
    rectangle_centered,
    plot_unit_cell,
    plot_array,
    export_to_gds
)

def main():
    # Example 1: L-shaped polygon
    print("Creating L-shaped polygon...")
    l_vertices = np.array([
        [0, 0],
        [60, 0],
        [60, 20],
        [20, 20],
        [20, 60],
        [0, 60]
    ], dtype=np.float64)
    
    # Center the L-shape
    centroid = l_vertices.mean(axis=0)
    l_vertices -= centroid
    
    l_poly = Polygon(vertices=l_vertices)
    print(f"L-shape area: {l_poly.area():.1f} nm²")
    print(f"L-shape centroid: {l_poly.centroid()}")
    
    # Create unit cell with L-shape
    unit_cell_l = UnitCell(
        pitch_x=150.0,
        pitch_y=150.0,
        polygons=(l_poly,)
    )
    
    # Visualize unit cell
    print("\nVisualizing L-shaped unit cell...")
    plot_unit_cell(unit_cell_l, title="L-Shaped Unit Cell")
    
    # Create array
    array_l = Array(unit_cell=unit_cell_l, rows=5, cols=5)
    plot_array(array_l, title="L-Shaped Array")
    
    # Example 2: Cross pattern
    print("\nCreating cross pattern...")
    
    # Horizontal bar
    h_bar = rectangle_centered(0.0, 0.0, 80.0, 20.0)
    # Vertical bar
    v_bar = rectangle_centered(0.0, 0.0, 20.0, 80.0)
    
    unit_cell_cross = UnitCell(
        pitch_x=150.0,
        pitch_y=150.0,
        polygons=(h_bar, v_bar)
    )
    
    print(f"Cross pattern density: {unit_cell_cross.density():.2%}")
    
    # Visualize
    plot_unit_cell(unit_cell_cross, title="Cross Pattern Unit Cell")
    
    array_cross = Array(unit_cell=unit_cell_cross, rows=5, cols=5)
    plot_array(array_cross, title="Cross Pattern Array")
    
    # Export to GDS
    print("\nExporting to GDS...")
    export_to_gds(array_cross, 'cross_pattern.gds', layer=1)
    print("Saved to cross_pattern.gds")
    
    # Example 3: Asymmetric pattern
    print("\nCreating asymmetric pattern...")
    
    poly1 = rectangle_centered(-30.0, 30.0, 40.0, 40.0)
    poly2 = rectangle_centered(30.0, -30.0, 40.0, 40.0)
    poly3 = rectangle_centered(0.0, 0.0, 20.0, 20.0)
    
    unit_cell_asym = UnitCell(
        pitch_x=200.0,
        pitch_y=200.0,
        polygons=(poly1, poly2, poly3)
    )
    
    print(f"Asymmetric pattern: {unit_cell_asym.num_polygons()} polygons per cell")
    
    plot_unit_cell(unit_cell_asym, title="Asymmetric Pattern")
    
    array_asym = Array(unit_cell=unit_cell_asym, rows=4, cols=4)
    plot_array(array_asym, title="Asymmetric Array")
    
    print("\nClose all plot windows to exit...")

if __name__ == '__main__':
    main()
