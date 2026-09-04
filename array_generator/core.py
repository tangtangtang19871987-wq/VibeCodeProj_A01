"""
Core data structures for array generation.

This module defines the fundamental data structures for representing polygons,
unit cells, and periodic arrays.
"""

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import warnings


@dataclass(frozen=True)
class Polygon:
    """
    Immutable polygon representation in canonical form.
    
    Canonical Form:
    - Vertices in counter-clockwise (CCW) order
    - First vertex is lexicographically smallest (min x, then min y)
    - No duplicate consecutive vertices
    
    Attributes:
        vertices: Array of shape (N, 2) with vertex coordinates (nm)
    """
    vertices: np.ndarray  # shape (N, 2), dtype float64
    
    def __post_init__(self):
        """Normalize to canonical form and validate."""
        # Ensure immutability - make a copy
        vertices_copy = self.vertices.copy()
        object.__setattr__(self, 'vertices', vertices_copy)
        
        # Validate
        if len(self.vertices) < 3:
            raise ValueError("Polygon must have at least 3 vertices")
        
        # Normalize to canonical form
        vertices = self.vertices.copy()
        
        # Ensure CCW order
        if not self._is_ccw(vertices):
            vertices = vertices[::-1]
        
        # Rotate to start with lexicographically smallest vertex
        start_idx = self._find_canonical_start(vertices)
        vertices = np.roll(vertices, -start_idx, axis=0)
        
        # Update with normalized vertices
        object.__setattr__(self, 'vertices', vertices)
        
        # Make immutable
        self.vertices.flags.writeable = False
    
    @staticmethod
    def _is_ccw(vertices: np.ndarray) -> bool:
        """Check if vertices are in counter-clockwise order using signed area."""
        signed_area = 0.0
        n = len(vertices)
        for i in range(n):
            v0 = vertices[i]
            v1 = vertices[(i + 1) % n]
            signed_area += (v1[0] - v0[0]) * (v1[1] + v0[1])
        return signed_area < 0
    
    @staticmethod
    def _find_canonical_start(vertices: np.ndarray) -> int:
        """Find index of lexicographically smallest vertex."""
        min_idx = 0
        min_v = vertices[0]
        for i in range(1, len(vertices)):
            v = vertices[i]
            if (v[0] < min_v[0]) or (v[0] == min_v[0] and v[1] < min_v[1]):
                min_idx = i
                min_v = v
        return min_idx
    
    def bbox(self) -> Tuple[float, float, float, float]:
        """
        Compute bounding box.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        return (
            float(np.min(self.vertices[:, 0])),
            float(np.min(self.vertices[:, 1])),
            float(np.max(self.vertices[:, 0])),
            float(np.max(self.vertices[:, 1]))
        )
    
    def centroid(self) -> Tuple[float, float]:
        """
        Compute polygon centroid.
        
        Returns:
            (cx, cy)
        """
        return (
            float(np.mean(self.vertices[:, 0])),
            float(np.mean(self.vertices[:, 1]))
        )
    
    def area(self) -> float:
        """
        Compute polygon area using shoelace formula.
        
        Returns:
            Area in nm²
        """
        n = len(self.vertices)
        area = 0.0
        for i in range(n):
            v0 = self.vertices[i]
            v1 = self.vertices[(i + 1) % n]
            area += v0[0] * v1[1] - v1[0] * v0[1]
        return abs(area) / 2.0
    
    def translate(self, offset: Tuple[float, float]) -> 'Polygon':
        """
        Create a new polygon translated by offset.
        
        Args:
            offset: (dx, dy) translation vector
            
        Returns:
            New translated polygon
        """
        new_vertices = self.vertices + np.array(offset)
        return Polygon(vertices=new_vertices)


@dataclass(frozen=True)
class UnitCell:
    """
    Unit cell definition for periodic arrays.
    
    The unit cell is a rectangular spatial region of dimensions
    (pitch_x, pitch_y) that contains zero or more polygons.
    
    Coordinate System:
    - Origin at center of unit cell (0, 0)
    - X-axis right, Y-axis up
    - Bounds: [-pitch_x/2, pitch_x/2] × [-pitch_y/2, pitch_y/2]
    
    Attributes:
        pitch_x: Unit cell width (nm)
        pitch_y: Unit cell height (nm)
        polygons: Tuple of polygons within unit cell
    """
    pitch_x: float
    pitch_y: float
    polygons: Tuple[Polygon, ...]  # Immutable tuple
    
    def __post_init__(self):
        """Validate unit cell."""
        if self.pitch_x <= 0:
            raise ValueError(f"pitch_x must be positive, got {self.pitch_x}")
        if self.pitch_y <= 0:
            raise ValueError(f"pitch_y must be positive, got {self.pitch_y}")
        
        # Validate all polygons fit within bounds
        half_x = self.pitch_x / 2
        half_y = self.pitch_y / 2
        
        for i, poly in enumerate(self.polygons):
            min_x, min_y, max_x, max_y = poly.bbox()
            if min_x < -half_x or max_x > half_x:
                raise ValueError(
                    f"Polygon {i} exceeds unit cell X bounds "
                    f"[{-half_x}, {half_x}]: [{min_x}, {max_x}]"
                )
            if min_y < -half_y or max_y > half_y:
                raise ValueError(
                    f"Polygon {i} exceeds unit cell Y bounds "
                    f"[{-half_y}, {half_y}]: [{min_y}, {max_y}]"
                )
    
    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Get unit cell bounds.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        half_x = self.pitch_x / 2
        half_y = self.pitch_y / 2
        return (-half_x, -half_y, half_x, half_y)
    
    def num_polygons(self) -> int:
        """Get number of polygons in unit cell."""
        return len(self.polygons)
    
    def total_area(self) -> float:
        """Get total area of all polygons in unit cell."""
        return sum(poly.area() for poly in self.polygons)
    
    def density(self) -> float:
        """
        Compute feature density (polygon area / unit cell area).
        
        Returns:
            Density as fraction (0 to 1)
        """
        cell_area = self.pitch_x * self.pitch_y
        return self.total_area() / cell_area


@dataclass(frozen=True)
class Array:
    """
    Periodic array generated by tiling a unit cell.
    
    The array consists of M rows × N columns of unit cells.
    
    Attributes:
        unit_cell: Unit cell definition
        rows: Number of rows (M, Y direction)
        cols: Number of columns (N, X direction)
        origin: Bottom-left corner of array (default: (0, 0))
    """
    unit_cell: UnitCell
    rows: int
    cols: int
    origin: Tuple[float, float] = (0.0, 0.0)
    
    def __post_init__(self):
        """Validate array parameters."""
        if self.rows <= 0:
            raise ValueError(f"rows must be positive, got {self.rows}")
        if self.cols <= 0:
            raise ValueError(f"cols must be positive, got {self.cols}")
        
        # Warn if array is very large
        total_polygons = self.rows * self.cols * self.unit_cell.num_polygons()
        if total_polygons > 10_000_000:
            warnings.warn(
                f"Array will generate {total_polygons:,} polygons. "
                "This may consume significant memory."
            )
    
    def num_cells(self) -> int:
        """Get total number of unit cells."""
        return self.rows * self.cols
    
    def num_polygons(self) -> int:
        """Get total number of polygons in array."""
        return self.num_cells() * self.unit_cell.num_polygons()
    
    def bounds(self) -> Tuple[float, float, float, float]:
        """
        Get array bounding box.
        
        Returns:
            (min_x, min_y, max_x, max_y)
        """
        min_x = self.origin[0]
        min_y = self.origin[1]
        max_x = min_x + self.cols * self.unit_cell.pitch_x
        max_y = min_y + self.rows * self.unit_cell.pitch_y
        return (min_x, min_y, max_x, max_y)
    
    def get_cell_center(self, i: int, j: int) -> Tuple[float, float]:
        """
        Get center coordinates of unit cell (i, j).
        
        Args:
            i: Column index (0 to cols-1)
            j: Row index (0 to rows-1)
            
        Returns:
            (cx, cy) center coordinates
        """
        if not (0 <= i < self.cols):
            raise ValueError(f"Column index {i} out of range [0, {self.cols})")
        if not (0 <= j < self.rows):
            raise ValueError(f"Row index {j} out of range [0, {self.rows})")
        
        cx = self.origin[0] + (i + 0.5) * self.unit_cell.pitch_x
        cy = self.origin[1] + (j + 0.5) * self.unit_cell.pitch_y
        return (cx, cy)
    
    def get_all_cell_centers(self) -> np.ndarray:
        """
        Get centers of all unit cells.
        
        Returns:
            Array of shape (rows*cols, 2) with cell centers
        """
        centers = []
        for j in range(self.rows):
            for i in range(self.cols):
                centers.append(self.get_cell_center(i, j))
        return np.array(centers)
    
    def generate_polygons(self) -> List[Polygon]:
        """
        Generate all polygons in the array with absolute world coordinates.
        
        Returns:
            List of polygons (length = rows * cols * polygons_per_cell)
        """
        all_polygons = []
        
        for j in range(self.rows):
            for i in range(self.cols):
                cell_center = self.get_cell_center(i, j)
                
                # Translate each polygon in unit cell to world coordinates
                for poly in self.unit_cell.polygons:
                    world_poly = poly.translate(cell_center)
                    all_polygons.append(world_poly)
        
        return all_polygons
    
    def generate_polygons_vectorized(self) -> np.ndarray:
        """
        Generate all polygon vertices using vectorized operations.
        
        More efficient for large arrays.
        
        Returns:
            Array of shape (num_polygons, num_vertices, 2)
        """
        # Get all cell centers
        centers = self.get_all_cell_centers()  # shape (rows*cols, 2)
        
        # For each polygon in unit cell, broadcast to all cells
        all_vertices = []
        for poly in self.unit_cell.polygons:
            # poly.vertices shape: (num_verts, 2)
            # centers shape: (num_cells, 2)
            # Result shape: (num_cells, num_verts, 2)
            vertices = poly.vertices[np.newaxis, :, :] + centers[:, np.newaxis, :]
            all_vertices.append(vertices)
        
        # Concatenate along polygon axis
        return np.concatenate(all_vertices, axis=0)
