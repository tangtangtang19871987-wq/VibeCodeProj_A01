"""
Example: Simple Contact Array

This example demonstrates how to create a basic square contact array
with one contact per unit cell.
"""

from array_generator import (
    simple_contact_array,
    export_to_gds,
    export_polygon_centroids,
    plot_array
)

def main():
    # Create a 10×10 array of 50nm square contacts with 100nm pitch
    print("Creating simple contact array...")
    array = simple_contact_array(
        cd=50.0,        # Contact critical dimension (nm)
        pitch=100.0,    # Pitch in both X and Y (nm)
        rows=10,        # Number of rows
        cols=10         # Number of columns
    )
    
    # Print array information
    print(f"Array dimensions: {array.rows}×{array.cols}")
    print(f"Total cells: {array.num_cells()}")
    print(f"Total polygons: {array.num_polygons()}")
    print(f"Array bounds: {array.bounds()}")
    
    # Print unit cell information
    uc = array.unit_cell
    print(f"\nUnit cell pitch: {uc.pitch_x}×{uc.pitch_y} nm")
    print(f"Polygons per cell: {uc.num_polygons()}")
    print(f"Feature density: {uc.density():.2%}")
    
    # Export to GDS
    print("\nExporting to GDS...")
    export_to_gds(array, 'simple_array.gds', layer=1)
    print("Saved to simple_array.gds")
    
    # Export centroids
    print("\nExporting centroids...")
    polygons = array.generate_polygons()
    export_polygon_centroids(polygons, 'simple_array_centroids.csv')
    print("Saved to simple_array_centroids.csv")
    
    # Visualize
    print("\nGenerating visualization...")
    plot_array(array, show_grid=True)
    print("Close the plot window to continue...")
    
    # Show a subset of a larger array
    print("\nCreating larger array...")
    large_array = simple_contact_array(cd=50.0, pitch=100.0, rows=50, cols=50)
    print(f"Large array: {large_array.num_polygons()} polygons")
    
    print("\nPlotting subset of large array...")
    plot_array(large_array, max_cells=100)
    print("Close the plot window to exit...")

if __name__ == '__main__':
    main()
