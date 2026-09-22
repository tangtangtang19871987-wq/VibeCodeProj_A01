"""Parser for ICCAD 2013 CAD Contest ``.glp`` layout files.

Format (observed in the contest benchmark, S2.1)::

    BEGIN
    EQUIV  1  1000  MICRON  +X,+Y
    CNAME Temp_Top
    LEVEL M1
    CELL Temp_Top PRIME
       RECT N M1  <x> <y> <w> <h>
       PGON N M1  <x1> <y1> <x2> <y2> ...
    ENDMSG

Coordinates are integers in nanometres (``EQUIV 1 1000 MICRON`` => 1000 units per
micron => 1 unit = 1 nm). ``RECT`` gives lower-left corner plus width and height.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

Polygon = list[tuple[int, int]]


@dataclass
class Design:
    """A parsed layout in nanometre coordinates.

    Attributes
    ----------
    polygons:
        List of closed polygons; each is a list of ``(x, y)`` integer nm vertices.
    """

    polygons: list[Polygon] = field(default_factory=list)

    @classmethod
    def from_glp(cls, path: str) -> "Design":
        """Parse a ``.glp`` file. Raises ``ValueError`` on a malformed record."""
        polys: list[Polygon] = []
        with open(path, "r") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 7:
                    continue
                if parts[0] == "RECT":
                    x, y, w, h = (int(v) for v in parts[3:7])
                    polys.append([(x, y), (x, y + h), (x + w, y + h), (x + w, y)])
                elif parts[0] == "PGON":
                    coords = [int(v) for v in parts[3:]]
                    if len(coords) % 2:
                        raise ValueError(f"PGON with odd coordinate count in {path}")
                    polys.append(list(zip(coords[0::2], coords[1::2])))
        if not polys:
            raise ValueError(f"No RECT/PGON records found in {path}")
        return cls(polys)

    def bbox(self) -> tuple[int, int, int, int]:
        """Return ``(min_x, min_y, max_x, max_y)`` in nm."""
        xs = [p[0] for poly in self.polygons for p in poly]
        ys = [p[1] for poly in self.polygons for p in poly]
        return min(xs), min(ys), max(xs), max(ys)

    def centred_raster(self, size: int = 2048) -> np.ndarray:
        """Rasterise onto a ``size x size`` canvas at 1 nm/pixel, bbox-centred.

        Returns
        -------
        np.ndarray
            ``(size, size)`` float32 array of ``{0.0, 1.0}``, indexed ``[row, col]``
            i.e. ``[y, x]``, matching the reference checker's convention.

        Raises
        ------
        ValueError
            If the layout does not fit in the canvas.
        """
        import cv2

        min_x, min_y, max_x, max_y = self.bbox()
        w, h = max_x - min_x, max_y - min_y
        if w > size or h > size:
            raise ValueError(f"Layout {w}x{h} nm exceeds {size} nm canvas")
        # Centre the bounding box on the canvas.
        off_x = (size - w) // 2 - min_x
        off_y = (size - h) // 2 - min_y

        canvas = np.zeros((size, size), dtype=np.float32)
        for poly in self.polygons:
            pts = np.array([[x + off_x, y + off_y] for x, y in poly], dtype=np.int32)
            cv2.fillPoly(canvas, [pts], 1.0)
        return canvas
