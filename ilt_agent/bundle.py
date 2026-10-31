"""LayoutBundle: one clip's target, mask and resist results on a common pixel grid.

Everything the analyzers need lives here so tools never touch the simulator:
  target   bool  [H, W]  design intent (True = drawn feature)
  mask     float [H, W]  mask transmission/coverage in [0, 1]
  printed  dict[str, bool [H, W]]   printed resist pattern per process condition
  intensity  float [H, W] | None    nominal resist image normalised so that
                                     printing happens where intensity > 1
  pixel_nm  float  pixel size in nm (wafer scale)
  nominal   str    name of the nominal condition inside `printed`
  meta      dict   free-form provenance (simulator, recipe, spec values ...)
Coordinates reported by every tool are (x_nm, y_nm) with the origin at the
lower-left pixel corner, x along columns, y along rows.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np


@dataclass
class LayoutBundle:
    target: np.ndarray
    mask: np.ndarray
    printed: dict
    pixel_nm: float
    nominal: str = "nominal"
    intensity: np.ndarray | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.target = np.asarray(self.target).astype(bool)
        self.mask = np.asarray(self.mask, dtype=np.float32)
        self.printed = {k: np.asarray(v).astype(bool) for k, v in self.printed.items()}
        if self.nominal not in self.printed:
            raise ValueError(f"nominal condition '{self.nominal}' not in printed: {list(self.printed)}")
        shapes = {self.target.shape, self.mask.shape, *(v.shape for v in self.printed.values())}
        if len(shapes) != 1:
            raise ValueError(f"all rasters must share one shape, got {shapes}")

    @property
    def shape(self):
        return self.target.shape

    @property
    def extent_nm(self):
        h, w = self.shape
        return w * self.pixel_nm, h * self.pixel_nm

    @property
    def conditions(self):
        return list(self.printed)

    def to_px(self, x_nm, y_nm):
        return int(round(y_nm / self.pixel_nm - 0.5)), int(round(x_nm / self.pixel_nm - 0.5))

    def to_nm(self, row, col):
        return (col + 0.5) * self.pixel_nm, (row + 0.5) * self.pixel_nm

    # ------------------------------------------------------------------ io
    def save(self, path):
        arrays = {"target": self.target, "mask": self.mask}
        for i, (k, v) in enumerate(self.printed.items()):
            arrays[f"printed_{i}"] = v
        if self.intensity is not None:
            arrays["intensity"] = self.intensity.astype(np.float32)
        header = {"pixel_nm": self.pixel_nm, "nominal": self.nominal,
                  "conditions": list(self.printed), "meta": self.meta}
        np.savez_compressed(path, header=np.array(json.dumps(header)), **arrays)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        h = json.loads(str(z["header"]))
        printed = {name: z[f"printed_{i}"] for i, name in enumerate(h["conditions"])}
        return cls(target=z["target"], mask=z["mask"], printed=printed, pixel_nm=float(h["pixel_nm"]),
                   nominal=h["nominal"], intensity=z["intensity"] if "intensity" in z else None,
                   meta=h.get("meta", {}))
