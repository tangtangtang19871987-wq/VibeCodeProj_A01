"""
Unit tests for constructor functions.
"""

import pytest
import numpy as np
from array_generator.constructors import rectangle_centered, rectangle_corner
from array_generator.core import Polygon


class TestRectangleCentered:
    """Tests for rectangle_centered function."""
    
    def test_basic_rectangle(self):
        """Test creating a basic centered rectangle."""
        rect = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        
        assert isinstance(rect, Polygon)
        min_x, min_y, max_x, max_y = rect.bbox()
        assert min_x == -50.0
        assert max_x == 50.0
        assert min_y == -25.0
        assert max_y == 25.0
    
    def test_offset_center(self):
        """Test rectangle with non-zero center."""
        rect = rectangle_centered(10.0, 20.0, 100.0, 50.0)
        
        min_x, min_y, max_x, max_y = rect.bbox()
        assert min_x == -40.0
        assert max_x == 60.0
        assert min_y == -5.0
        assert max_y == 45.0
    
    def test_square(self):
        """Test creating a square."""
        rect = rectangle_centered(0.0, 0.0, 100.0, 100.0)
        
        min_x, min_y, max_x, max_y = rect.bbox()
        assert max_x - min_x == 100.0
        assert max_y - min_y == 100.0
    
    def test_canonical_form(self):
        """Test that rectangle is in canonical form."""
        rect = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        
        # First vertex should be lexicographically smallest
        first = rect.vertices[0]
        for v in rect.vertices[1:]:
            assert (first[0] < v[0]) or (first[0] == v[0] and first[1] <= v[1])
    
    def test_area(self):
        """Test that rectangle area is correct."""
        rect = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        assert rect.area() == 5000.0
    
    def test_centroid(self):
        """Test that centroid is at specified center."""
        cx, cy = 10.0, 20.0
        rect = rectangle_centered(cx, cy, 100.0, 50.0)
        
        centroid = rect.centroid()
        assert abs(centroid[0] - cx) < 1e-10
        assert abs(centroid[1] - cy) < 1e-10
    
    def test_negative_width(self):
        """Test that negative width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            rectangle_centered(0.0, 0.0, -100.0, 50.0)
    
    def test_zero_width(self):
        """Test that zero width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            rectangle_centered(0.0, 0.0, 0.0, 50.0)
    
    def test_negative_height(self):
        """Test that negative height raises ValueError."""
        with pytest.raises(ValueError, match="height must be positive"):
            rectangle_centered(0.0, 0.0, 100.0, -50.0)
    
    def test_zero_height(self):
        """Test that zero height raises ValueError."""
        with pytest.raises(ValueError, match="height must be positive"):
            rectangle_centered(0.0, 0.0, 100.0, 0.0)
    
    def test_very_small_rectangle(self):
        """Test creating a very small rectangle."""
        rect = rectangle_centered(0.0, 0.0, 1.0, 1.0)
        assert rect.area() == 1.0
    
    def test_very_large_rectangle(self):
        """Test creating a very large rectangle."""
        rect = rectangle_centered(0.0, 0.0, 1e6, 1e6)
        assert rect.area() == 1e12


class TestRectangleCorner:
    """Tests for rectangle_corner function."""
    
    def test_basic_rectangle(self):
        """Test creating a basic corner-based rectangle."""
        rect = rectangle_corner(0.0, 0.0, 100.0, 50.0)
        
        assert isinstance(rect, Polygon)
        min_x, min_y, max_x, max_y = rect.bbox()
        assert min_x == 0.0
        assert max_x == 100.0
        assert min_y == 0.0
        assert max_y == 50.0
    
    def test_offset_corner(self):
        """Test rectangle with non-zero corner."""
        rect = rectangle_corner(10.0, 20.0, 100.0, 50.0)
        
        min_x, min_y, max_x, max_y = rect.bbox()
        assert min_x == 10.0
        assert max_x == 110.0
        assert min_y == 20.0
        assert max_y == 70.0
    
    def test_negative_corner(self):
        """Test rectangle with negative corner coordinates."""
        rect = rectangle_corner(-50.0, -25.0, 100.0, 50.0)
        
        min_x, min_y, max_x, max_y = rect.bbox()
        assert min_x == -50.0
        assert max_x == 50.0
        assert min_y == -25.0
        assert max_y == 25.0
    
    def test_area(self):
        """Test that rectangle area is correct."""
        rect = rectangle_corner(0.0, 0.0, 100.0, 50.0)
        assert rect.area() == 5000.0
    
    def test_centroid(self):
        """Test that centroid is correctly positioned."""
        rect = rectangle_corner(0.0, 0.0, 100.0, 50.0)
        
        centroid = rect.centroid()
        assert abs(centroid[0] - 50.0) < 1e-10
        assert abs(centroid[1] - 25.0) < 1e-10
    
    def test_equivalence_to_centered(self):
        """Test that corner and centered produce equivalent rectangles."""
        corner_rect = rectangle_corner(0.0, 0.0, 100.0, 50.0)
        centered_rect = rectangle_centered(50.0, 25.0, 100.0, 50.0)
        
        # Should have same bbox
        assert corner_rect.bbox() == centered_rect.bbox()
        
        # Should have same area
        assert corner_rect.area() == centered_rect.area()
    
    def test_negative_width(self):
        """Test that negative width raises ValueError."""
        with pytest.raises(ValueError, match="width must be positive"):
            rectangle_corner(0.0, 0.0, -100.0, 50.0)
    
    def test_negative_height(self):
        """Test that negative height raises ValueError."""
        with pytest.raises(ValueError, match="height must be positive"):
            rectangle_corner(0.0, 0.0, 100.0, -50.0)
