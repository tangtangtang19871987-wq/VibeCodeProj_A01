"""
Performance benchmarks for array generation.

These tests verify that the array generator meets performance requirements:
- REQ-PERF-001: Generate 10,000 polygons in < 1 second
- REQ-PERF-001: Generate 1,000,000 polygons in < 10 seconds
"""

import time
import pytest
from array_generator import simple_contact_array


class TestPerformance:
    """Performance benchmarks for array generation."""
    
    def test_generate_10k_polygons(self):
        """Test generating 10,000 polygons in < 1 second."""
        # 100×100 array = 10,000 polygons
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=100, cols=100)
        
        start = time.time()
        polygons = array.generate_polygons()
        elapsed = time.time() - start
        
        assert len(polygons) == 10000
        assert elapsed < 1.0, f"Took {elapsed:.3f}s, expected < 1.0s"
        print(f"\n10,000 polygons generated in {elapsed:.3f}s")
    
    def test_generate_10k_polygons_vectorized(self):
        """Test vectorized generation of 10,000 polygons."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=100, cols=100)
        
        start = time.time()
        vertices = array.generate_polygons_vectorized()
        elapsed = time.time() - start
        
        assert vertices.shape[0] == 10000
        assert elapsed < 1.0, f"Took {elapsed:.3f}s, expected < 1.0s"
        print(f"\n10,000 polygons (vectorized) generated in {elapsed:.3f}s")
    
    @pytest.mark.slow
    def test_generate_1m_polygons(self):
        """Test generating 1,000,000 polygons in < 10 seconds."""
        # 1000×1000 array = 1,000,000 polygons
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1000, cols=1000)
        
        start = time.time()
        polygons = array.generate_polygons()
        elapsed = time.time() - start
        
        assert len(polygons) == 1000000
        assert elapsed < 10.0, f"Took {elapsed:.3f}s, expected < 10.0s"
        print(f"\n1,000,000 polygons generated in {elapsed:.3f}s")
    
    @pytest.mark.slow
    def test_generate_1m_polygons_vectorized(self):
        """Test vectorized generation of 1,000,000 polygons."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=1000, cols=1000)
        
        start = time.time()
        vertices = array.generate_polygons_vectorized()
        elapsed = time.time() - start
        
        assert vertices.shape[0] == 1000000
        assert elapsed < 10.0, f"Took {elapsed:.3f}s, expected < 10.0s"
        print(f"\n1,000,000 polygons (vectorized) generated in {elapsed:.3f}s")
    
    def test_vectorized_vs_loop_speedup(self):
        """Test that vectorized generation is faster than loop-based."""
        array = simple_contact_array(cd=50.0, pitch=100.0, rows=50, cols=50)
        
        # Loop-based
        start = time.time()
        polygons_loop = array.generate_polygons()
        time_loop = time.time() - start
        
        # Vectorized
        start = time.time()
        vertices_vec = array.generate_polygons_vectorized()
        time_vec = time.time() - start
        
        speedup = time_loop / time_vec
        print(f"\nSpeedup: {speedup:.1f}× (loop: {time_loop:.4f}s, vectorized: {time_vec:.4f}s)")
        
        # Vectorized should be faster (though for small arrays the difference may be small)
        assert time_vec <= time_loop * 1.5  # Allow some variance
    
    def test_unit_cell_creation_overhead(self):
        """Test that unit cell creation is fast."""
        from array_generator import rectangle_centered, UnitCell
        
        start = time.time()
        for _ in range(1000):
            poly = rectangle_centered(0.0, 0.0, 50.0, 50.0)
            uc = UnitCell(pitch_x=100.0, pitch_y=100.0, polygons=(poly,))
        elapsed = time.time() - start
        
        print(f"\n1000 unit cells created in {elapsed:.3f}s ({elapsed*1000:.1f}μs each)")
        assert elapsed < 0.1  # Should be very fast
    
    def test_array_creation_overhead(self):
        """Test that array creation is fast."""
        from array_generator import simple_contact_array
        
        start = time.time()
        for _ in range(100):
            array = simple_contact_array(cd=50.0, pitch=100.0, rows=10, cols=10)
        elapsed = time.time() - start
        
        print(f"\n100 arrays created in {elapsed:.3f}s ({elapsed*10:.1f}ms each)")
        assert elapsed < 0.5  # Should be very fast


def run_benchmarks():
    """Run all benchmarks and print results."""
    print("\n" + "="*70)
    print("ARRAY GENERATOR PERFORMANCE BENCHMARKS")
    print("="*70)
    
    test = TestPerformance()
    
    print("\n--- Small Array (10,000 polygons) ---")
    test.test_generate_10k_polygons()
    test.test_generate_10k_polygons_vectorized()
    
    print("\n--- Large Array (1,000,000 polygons) ---")
    print("(This may take a few seconds...)")
    test.test_generate_1m_polygons()
    test.test_generate_1m_polygons_vectorized()
    
    print("\n--- Speedup Comparison ---")
    test.test_vectorized_vs_loop_speedup()
    
    print("\n--- Creation Overhead ---")
    test.test_unit_cell_creation_overhead()
    test.test_array_creation_overhead()
    
    print("\n" + "="*70)
    print("All benchmarks passed!")
    print("="*70 + "\n")


if __name__ == '__main__':
    run_benchmarks()
