"""
Unit tests for pattern templates.
"""

import pytest
from array_generator.templates import (
    simple_contact_array,
    staggered_contact_array,
    cluster_contact_array
)
from array_generator.core import Array


class TestSimpleContactArray:
    """Tests for simple_contact_array template."""
    
    def test_basic_array(self):
        """Test creating a basic simple contact array."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=10, cols=10)
        
        assert isinstance(array, Array)
        assert array.rows == 10
        assert array.cols == 10
        assert array.num_cells() == 100
        assert array.num_polygons() == 100  # 1 per cell
    
    def test_unit_cell_properties(self):
        """Test that unit cell has correct properties."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=5, cols=5)
        
        uc = array.unit_cell
        assert uc.pitch_x == 100.0
        assert uc.pitch_y == 100.0
        assert uc.num_polygons() == 1
    
    def test_polygon_size(self):
        """Test that contact has correct size."""
        cd = 50.0
        array = simple_contact_array(cd=cd, pitch=100.0, rows=1, cols=1)
        
        poly = array.unit_cell.polygons[0]
        min_x, min_y, max_x, max_y = poly.bbox()
        assert max_x - min_x == cd
        assert max_y - min_y == cd
    
    def test_different_dimensions(self):
        """Test with different row and column counts."""
        array = simple_contact_array(cd=30.0, pitch=100.0, rows=5, cols=8)
        
        assert array.rows == 5
        assert array.cols == 8
        assert array.num_polygons() == 40
    
    def test_custom_origin(self):
        """Test with custom origin."""
        origin = (100.0, 200.0)
        array = simple_contact_array(
            cd=50.0, pitch=100.0, rows=5, cols=5, origin=origin
        )
        
        assert array.origin == origin
        min_x, min_y, max_x, max_y = array.bounds()
        assert min_x == origin[0]
        assert min_y == origin[1]
    
    def test_cd_equals_pitch(self):
        """Test that cd == pitch raises ValueError."""
        with pytest.raises(ValueError, match="cd .* must be less than pitch"):
            simple_contact_array(cd=100.0, pitch=100.0, rows=5, cols=5)
    
    def test_cd_greater_than_pitch(self):
        """Test that cd > pitch raises ValueError."""
        with pytest.raises(ValueError, match="cd .* must be less than pitch"):
            simple_contact_array(cd=150.0, pitch=100.0, rows=5, cols=5)
    
    def test_small_array(self):
        """Test 1x1 array."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1, cols=1)
        assert array.num_polygons() == 1
    
    def test_large_array(self):
        """Test large array."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=100, cols=100)
        assert array.num_polygons() == 10000


class TestStaggeredContactArray:
    """Tests for staggered_contact_array template."""
    
    def test_basic_array(self):
        """Test creating a basic staggered array."""
        array = staggered_contact_array(
            cd=50.0, pitch_x=100.0, pitch_y=100.0, rows=10, cols=10
        )
        
        assert isinstance(array, Array)
        assert array.rows == 10
        assert array.cols == 10
        assert array.num_polygons() == 200  # 2 per cell
    
    def test_unit_cell_properties(self):
        """Test that unit cell has correct properties."""
        array = staggered_contact_array(
            cd=50.0, pitch_x=100.0, pitch_y=120.0, rows=5, cols=5
        )
        
        uc = array.unit_cell
        assert uc.pitch_x == 100.0
        assert uc.pitch_y == 120.0
        assert uc.num_polygons() == 2
    
    def test_stagger_offset(self):
        """Test that contacts are staggered correctly."""
        stagger_fraction = 0.25
        pitch_y = 100.0
        array = staggered_contact_array(
            cd=30.0,
            pitch_x=100.0,
            pitch_y=pitch_y,
            rows=1,
            cols=1,
            stagger_fraction=stagger_fraction
        )
        
        poly1, poly2 = array.unit_cell.polygons
        cy1 = poly1.centroid()[1]
        cy2 = poly2.centroid()[1]
        
        # One should be at +offset, one at -offset
        expected_offset = pitch_y * stagger_fraction
        assert abs(abs(cy1) - expected_offset) < 1e-10
        assert abs(abs(cy2) - expected_offset) < 1e-10
        assert cy1 == -cy2  # Symmetric about center
    
    def test_custom_stagger_fraction(self):
        """Test with custom stagger fraction."""
        array = staggered_contact_array(
            cd=30.0,
            pitch_x=100.0,
            pitch_y=100.0,
            rows=5,
            cols=5,
            stagger_fraction=0.3
        )
        
        assert array.num_polygons() == 50
    
    def test_different_pitches(self):
        """Test with different X and Y pitches."""
        array = staggered_contact_array(
            cd=30.0, pitch_x=100.0, pitch_y=150.0, rows=5, cols=8
        )
        
        assert array.unit_cell.pitch_x == 100.0
        assert array.unit_cell.pitch_y == 150.0
    
    def test_custom_origin(self):
        """Test with custom origin."""
        origin = (50.0, 75.0)
        array = staggered_contact_array(
            cd=30.0,
            pitch_x=100.0,
            pitch_y=100.0,
            rows=5,
            cols=5,
            origin=origin
        )
        
        assert array.origin == origin
    
    def test_cd_too_large(self):
        """Test that cd >= min(pitch_x, pitch_y) raises ValueError."""
        with pytest.raises(ValueError, match="cd .* must be less than"):
            staggered_contact_array(
                cd=100.0, pitch_x=100.0, pitch_y=150.0, rows=5, cols=5
            )


class TestClusterContactArray:
    """Tests for cluster_contact_array template."""
    
    def test_basic_array(self):
        """Test creating a basic cluster array."""
        array = cluster_contact_array(
            cd=30.0, spacing=50.0, pitch=200.0, rows=5, cols=5, cluster_size=3
        )
        
        assert isinstance(array, Array)
        assert array.rows == 5
        assert array.cols == 5
        assert array.num_polygons() == 75  # 3 per cell * 25 cells
    
    def test_unit_cell_properties(self):
        """Test that unit cell has correct properties."""
        cluster_size = 4
        array = cluster_contact_array(
            cd=20.0, spacing=40.0, pitch=200.0, rows=5, cols=5, cluster_size=cluster_size
        )
        
        uc = array.unit_cell
        assert uc.pitch_x == 200.0
        assert uc.pitch_y == 200.0
        assert uc.num_polygons() == cluster_size
    
    def test_cluster_spacing(self):
        """Test that contacts are spaced correctly."""
        cd = 20.0
        spacing = 50.0
        cluster_size = 3
        array = cluster_contact_array(
            cd=cd, spacing=spacing, pitch=200.0, rows=1, cols=1, cluster_size=cluster_size
        )
        
        polygons = array.unit_cell.polygons
        centroids = [p.centroid()[0] for p in polygons]
        centroids.sort()
        
        # Check spacing between consecutive contacts
        for i in range(len(centroids) - 1):
            assert abs(centroids[i+1] - centroids[i] - spacing) < 1e-10
    
    def test_cluster_centered(self):
        """Test that cluster is centered in unit cell."""
        array = cluster_contact_array(
            cd=20.0, spacing=50.0, pitch=200.0, rows=1, cols=1, cluster_size=3
        )
        
        polygons = array.unit_cell.polygons
        centroids = [p.centroid()[0] for p in polygons]
        
        # Mean of centroids should be near 0
        mean_x = sum(centroids) / len(centroids)
        assert abs(mean_x) < 1e-10
    
    def test_single_contact_cluster(self):
        """Test cluster with size 1."""
        array = cluster_contact_array(
            cd=30.0, spacing=50.0, pitch=200.0, rows=5, cols=5, cluster_size=1
        )
        
        assert array.num_polygons() == 25  # 1 per cell
    
    def test_large_cluster(self):
        """Test cluster with many contacts."""
        cluster_size = 10
        array = cluster_contact_array(
            cd=10.0, spacing=20.0, pitch=300.0, rows=3, cols=3, cluster_size=cluster_size
        )
        
        assert array.num_polygons() == 90  # 10 per cell * 9 cells
    
    def test_custom_origin(self):
        """Test with custom origin."""
        origin = (100.0, 150.0)
        array = cluster_contact_array(
            cd=20.0,
            spacing=40.0,
            pitch=200.0,
            rows=5,
            cols=5,
            cluster_size=3,
            origin=origin
        )
        
        assert array.origin == origin
    
    def test_cluster_too_wide(self):
        """Test that cluster wider than pitch raises ValueError."""
        with pytest.raises(ValueError, match="Cluster width .* must be less than pitch"):
            cluster_contact_array(
                cd=30.0, spacing=100.0, pitch=200.0, rows=5, cols=5, cluster_size=3
            )
    
    def test_zero_cluster_size(self):
        """Test that cluster_size = 0 raises ValueError."""
        with pytest.raises(ValueError, match="cluster_size must be >= 1"):
            cluster_contact_array(
                cd=20.0, spacing=40.0, pitch=200.0, rows=5, cols=5, cluster_size=0
            )
    
    def test_negative_cluster_size(self):
        """Test that negative cluster_size raises ValueError."""
        with pytest.raises(ValueError, match="cluster_size must be >= 1"):
            cluster_contact_array(
                cd=20.0, spacing=40.0, pitch=200.0, rows=5, cols=5, cluster_size=-1
            )
