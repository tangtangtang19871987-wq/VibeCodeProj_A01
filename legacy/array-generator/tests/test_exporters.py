"""
Unit tests for export functions.
"""

import pytest
import os
import tempfile
from array_generator.exporters import (
    export_polygon_vertices,
    export_polygon_centroids,
    export_to_gds,
    export_array_metadata
)
from array_generator.constructors import rectangle_centered
from array_generator.templates import simple_contact_array


class TestExportPolygonVertices:
    """Tests for export_polygon_vertices function."""
    
    def test_csv_export(self):
        """Test exporting vertices in CSV format."""
        poly = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            filename = f.name
        
        try:
            export_polygon_vertices([poly], filename, format='csv')
            
            # Read back and verify
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Should have header + 4 vertices
            assert len(lines) >= 6  # 2 header lines + 4 vertices
            
            # Check that commas are used
            data_lines = [l for l in lines if not l.startswith('#')]
            assert all(',' in line for line in data_lines)
            
        finally:
            os.unlink(filename)
    
    def test_space_export(self):
        """Test exporting vertices in space-separated format."""
        poly = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filename = f.name
        
        try:
            export_polygon_vertices([poly], filename, format='space')
            
            # Read back and verify
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Should have header + 4 vertices
            assert len(lines) >= 6
            
        finally:
            os.unlink(filename)
    
    def test_multiple_polygons(self):
        """Test exporting multiple polygons."""
        poly1 = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        poly2 = rectangle_centered(200.0, 0.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            filename = f.name
        
        try:
            export_polygon_vertices([poly1, poly2], filename, format='csv')
            
            # Read back and verify
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Should have header + 8 vertices (4 per polygon)
            data_lines = [l for l in lines if not l.startswith('#')]
            assert len(data_lines) == 8
            
        finally:
            os.unlink(filename)
    
    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        poly = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            filename = f.name
        
        try:
            with pytest.raises(ValueError, match="format must be"):
                export_polygon_vertices([poly], filename, format='invalid')
        finally:
            os.unlink(filename)


class TestExportPolygonCentroids:
    """Tests for export_polygon_centroids function."""
    
    def test_csv_export(self):
        """Test exporting centroids in CSV format."""
        poly = rectangle_centered(10.0, 20.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            filename = f.name
        
        try:
            export_polygon_centroids([poly], filename, format='csv')
            
            # Read back and verify
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            # Should have header + 1 centroid
            assert len(lines) >= 3
            
            # Check that commas are used
            data_lines = [l for l in lines if not l.startswith('#')]
            assert all(',' in line for line in data_lines)
            
            # Check centroid values
            data = data_lines[0].strip().split(',')
            assert len(data) == 3  # poly_id, cx, cy
            assert float(data[1]) == 10.0
            assert float(data[2]) == 20.0
            
        finally:
            os.unlink(filename)
    
    def test_space_export(self):
        """Test exporting centroids in space-separated format."""
        poly = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filename = f.name
        
        try:
            export_polygon_centroids([poly], filename, format='space')
            
            # Read back and verify
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            assert len(lines) >= 3
            
        finally:
            os.unlink(filename)
    
    def test_multiple_polygons(self):
        """Test exporting centroids of multiple polygons."""
        poly1 = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        poly2 = rectangle_centered(200.0, 100.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as f:
            filename = f.name
        
        try:
            export_polygon_centroids([poly1, poly2], filename, format='csv')
            
            # Read back and verify
            with open(filename, 'r') as f:
                lines = f.readlines()
            
            data_lines = [l for l in lines if not l.startswith('#')]
            assert len(data_lines) == 2
            
        finally:
            os.unlink(filename)
    
    def test_invalid_format(self):
        """Test that invalid format raises ValueError."""
        poly = rectangle_centered(0.0, 0.0, 100.0, 50.0)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            filename = f.name
        
        try:
            with pytest.raises(ValueError, match="format must be"):
                export_polygon_centroids([poly], filename, format='json')
        finally:
            os.unlink(filename)


class TestExportToGDS:
    """Tests for export_to_gds function."""
    
    def test_basic_export(self):
        """Test basic GDS export."""
        try:
            import gdspy
        except ImportError:
            pytest.skip("gdspy not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=2, cols=2)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.gds') as f:
            filename = f.name
        
        try:
            export_to_gds(array, filename, layer=1)
            
            # Verify file was created
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
            
            # Try to read it back
            lib = gdspy.GdsLibrary(infile=filename)
            assert 'ARRAY' in lib.cells
            
        finally:
            os.unlink(filename)
    
    def test_custom_layer(self):
        """Test GDS export with custom layer."""
        try:
            import gdspy
        except ImportError:
            pytest.skip("gdspy not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1, cols=1)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.gds') as f:
            filename = f.name
        
        try:
            export_to_gds(array, filename, layer=5, datatype=2)
            
            # Read back and check layer
            lib = gdspy.GdsLibrary(infile=filename)
            cell = lib.cells['ARRAY']
            
            # Check that polygons exist
            assert len(cell.polygons) > 0
            
        finally:
            os.unlink(filename)
    
    def test_large_array(self):
        """Test GDS export with larger array."""
        try:
            import gdspy
        except ImportError:
            pytest.skip("gdspy not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=10, cols=10)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.gds') as f:
            filename = f.name
        
        try:
            export_to_gds(array, filename)
            
            # Read back and verify polygon count
            lib = gdspy.GdsLibrary(infile=filename)
            cell = lib.cells['ARRAY']
            assert len(cell.polygons) == 100  # 10x10 array
            
        finally:
            os.unlink(filename)


class TestExportArrayMetadata:
    """Tests for export_array_metadata function."""
    
    def test_basic_metadata(self):
        """Test exporting array metadata."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=5, cols=8)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filename = f.name
        
        try:
            export_array_metadata(array, filename)
            
            # Read back and verify
            with open(filename, 'r') as f:
                content = f.read()
            
            # Check that key information is present
            assert 'pitch_x: 100.0' in content
            assert 'pitch_y: 100.0' in content
            assert 'rows: 5' in content
            assert 'cols: 8' in content
            assert 'total_cells: 40' in content
            assert 'total_polygons: 40' in content
            
        finally:
            os.unlink(filename)
    
    def test_metadata_with_custom_origin(self):
        """Test metadata export with custom origin."""
        array = simple_contact_array(
            cd=50.0, pitch=100.0, rows=3, cols=3, origin=(100.0, 200.0)
        )
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
            filename = f.name
        
        try:
            export_array_metadata(array, filename)
            
            with open(filename, 'r') as f:
                content = f.read()
            
            assert 'origin: (100.0, 200.0)' in content
            
        finally:
            os.unlink(filename)
