"""Unit tests for the ICCAD13 GLP layout parser."""
import numpy as np
import pytest

from gril.data.glp import Design


def test_parses_rect_records(tmp_path):
    f = tmp_path / "t.glp"
    f.write_text(
        "BEGIN\nEQUIV  1  1000  MICRON  +X,+Y\nCELL Temp_Top PRIME\n"
        "   RECT N M1  10  20  30  40\nENDMSG\n"
    )
    d = Design.from_glp(str(f))
    assert len(d.polygons) == 1
    assert d.bbox() == (10, 20, 40, 60)          # x+w, y+h


def test_parses_pgon_records(tmp_path):
    f = tmp_path / "t.glp"
    f.write_text(
        "CELL Temp_Top PRIME\n   PGON N M1  0 0  0 10  10 10  10 0\nENDMSG\n"
    )
    d = Design.from_glp(str(f))
    assert d.polygons[0] == [(0, 0), (0, 10), (10, 10), (10, 0)]


def test_rejects_file_without_shapes(tmp_path):
    f = tmp_path / "empty.glp"
    f.write_text("BEGIN\nENDMSG\n")
    with pytest.raises(ValueError):
        Design.from_glp(str(f))


def test_raster_area_matches_rect(tmp_path):
    f = tmp_path / "t.glp"
    f.write_text("CELL X PRIME\n   RECT N M1  0  0  100  50\nENDMSG\n")
    img = Design.from_glp(str(f)).centred_raster(512)
    assert img.shape == (512, 512)
    assert img.dtype == np.float32
    assert set(np.unique(img)) <= {0.0, 1.0}
    # Polygon fill is boundary-INCLUSIVE: a w x h RECT occupies (w+1) x (h+1)
    # pixels. This is the published ICCAD13 convention (verified bit-identical
    # against the reference rasteriser) and is deliberately preserved, because
    # every contest number depends on it. See REPRODUCTION_SPEC.md Sec. 2.
    assert img.sum() == 101 * 51


def test_raster_is_centred(tmp_path):
    f = tmp_path / "t.glp"
    f.write_text("CELL X PRIME\n   RECT N M1  500  700  100  100\nENDMSG\n")
    img = Design.from_glp(str(f)).centred_raster(512)
    rows = np.nonzero(img.sum(1))[0]
    cols = np.nonzero(img.sum(0))[0]
    # Centre of mass should sit at the canvas centre to within a pixel.
    assert abs((rows.min() + rows.max()) / 2 - 255.5) <= 1.5
    assert abs((cols.min() + cols.max()) / 2 - 255.5) <= 1.5


def test_raster_rejects_oversized_layout(tmp_path):
    f = tmp_path / "t.glp"
    f.write_text("CELL X PRIME\n   RECT N M1  0  0  5000  50\nENDMSG\n")
    with pytest.raises(ValueError):
        Design.from_glp(str(f)).centred_raster(2048)
