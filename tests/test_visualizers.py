"""
Unit tests for visualization functions.
"""

import pytest
import os
import tempfile
from array_generator.visualizers import (
    plot_unit_cell,
    plot_array,
    save_unit_cell_plot,
    save_array_plot
)
from array_generator.templates import simple_contact_array, staggered_contact_array


class TestPlotUnitCell:
    """Tests for plot_unit_cell function."""
    
    def test_basic_plot(self):
        """Test that plotting doesn't crash."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1, cols=1)
        
        # Should not raise an exception
        plot_unit_cell(array.unit_cell)
    
    def test_plot_without_bounds(self):
        """Test plotting without bounds."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1, cols=1)
        plot_unit_cell(array.unit_cell, show_bounds=False)
    
    def test_plot_with_custom_title(self):
        """Test plotting with custom title."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1, cols=1)
        plot_unit_cell(array.unit_cell, title="Custom Title")
    
    def test_plot_multiple_polygons(self):
        """Test plotting unit cell with multiple polygons."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = staggered_contact_array(
            cd=30.0, pitch_x=100.0, pitch_y=100.0, rows=1, cols=1
        )
        plot_unit_cell(array.unit_cell)


class TestPlotArray:
    """Tests for plot_array function."""
    
    def test_basic_plot(self):
        """Test that plotting doesn't crash."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=5, cols=5)
        plot_array(array)
    
    def test_plot_without_grid(self):
        """Test plotting without grid."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=5, cols=5)
        plot_array(array, show_grid=False)
    
    def test_plot_with_custom_title(self):
        """Test plotting with custom title."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=5, cols=5)
        plot_array(array, title="Custom Array")
    
    def test_plot_large_array_subset(self):
        """Test plotting subset of large array."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        # Create array larger than max_cells
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=20, cols=20)
        plot_array(array, max_cells=50)
    
    def test_plot_small_array(self):
        """Test plotting small array."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=2, cols=2)
        plot_array(array)


class TestSaveUnitCellPlot:
    """Tests for save_unit_cell_plot function."""
    
    def test_save_plot(self):
        """Test saving unit cell plot to file."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1, cols=1)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.png') as f:
            filename = f.name
        
        try:
            save_unit_cell_plot(array.unit_cell, filename)
            
            # Verify file was created
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
            
        finally:
            os.unlink(filename)
    
    def test_save_plot_custom_dpi(self):
        """Test saving with custom DPI."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1, cols=1)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.png') as f:
            filename = f.name
        
        try:
            save_unit_cell_plot(array.unit_cell, filename, dpi=300)
            assert os.path.exists(filename)
            
        finally:
            os.unlink(filename)


class TestSaveArrayPlot:
    """Tests for save_array_plot function."""
    
    def test_save_plot(self):
        """Test saving array plot to file."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=5, cols=5)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.png') as f:
            filename = f.name
        
        try:
            save_array_plot(array, filename)
            
            # Verify file was created
            assert os.path.exists(filename)
            assert os.path.getsize(filename) > 0
            
        finally:
            os.unlink(filename)
    
    def test_save_plot_custom_dpi(self):
        """Test saving with custom DPI."""
        try:
            import matplotlib
            matplotlib.use('Agg')
        except ImportError:
            pytest.skip("matplotlib not installed")
        
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=5, cols=5)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.png') as f:
            filename = f.name
        
        try:
            save_array_plot(array, filename, dpi=300)
            assert os.path.exists(filename)
            
        finally:
            os.unlink(filename)
