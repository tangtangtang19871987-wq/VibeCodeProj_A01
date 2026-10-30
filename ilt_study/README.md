# ILT bridge/pinch recipe study (pitch 74.5 nm, 193i)

A small, fully specified, differentiable ILT framework built to test ILT
*recipe-level* claims about the bridge-vs-pinch conflict numerically.
The model is held fixed on purpose: the question is what the recipe
(cost, corners, freedoms, optimizer) can do on a given model.

## Model (litho.py)
* Scalar Abbe imaging via SOCS (13 TCC kernels, 0.06 % max error vs Abbe),
  lambda = 193 nm, NA = 1.35, immersion n = 1.44, exact defocus phase.
* Leaf-shaped X-dipole: sigma_x in [0.90, 1], |sigma_y| <= 0.30 -- the region
  where 0 and +-1 orders of a 74.5 nm pitch both pass the pupil.
* Resist: 5 nm Gaussian blur + constant threshold, calibrated so a 37/37.5
  dense grating prints on size with mask = target.
* Thin binary dark-field mask (bright = trench = metal line).
* 596 nm periodic clip (8 tracks), 256 px (2.33 nm/px).

## ILT (ilt.py)
* theta -> sigmoid -> cone filter (radius R, the MRC proxy) -> tanh projection.
* Mask-CD-error corners: rho + b |grad rho| (iso-contour moves by b nm).
* Corners (optimization): nominal + {dose 1.05, bias +1} and {dose 0.95,
  bias -1} at focus 0, +-40 nm.  Checker: full 3x3x3 factorial.
* Cost terms: image fidelity, global outer/inner PV band (area type),
  per-site hinge / quadratic / linear terms on cutline gauges (soft-min over
  corners), SRAF-print term, log-contrast term, keep-out/keep-in zones,
  per-site weights, caps and minimax groups, frozen-pixel regions.
* Adam (default) or plain gradient descent.

## Checker (layout.py, ilt.py)
* Cutline sites: pinch (line width >= 28), lateral bridge (space >= 28),
  tip-to-tip (gap >= 30, three parallel cuts), diagonal line-end-to-side
  (>= 28), line-end pull-back (<= 15).  Measured on the *hard* (sub-pixel
  anti-aliased) mask at all 27 corners.
* Topology: printed components merging two targets (bridge), broken /
  missing targets (open), printed components touching no target (SRAF print).
* `n_hot` = site violations + topological bridges + opens.

## Running
```
pip install torch numpy matplotlib
python3 experiments/e01_cosine_limit.py      # each experiment is standalone
./run_pipeline.sh                            # E5..E14 in sequence
# follow-ups: e05b, e07b, e07c, e15 (run_pipeline2..6.sh)
```
Results land in `results/*.json` (+ logs), figures in `figures/`.
A full run of all experiments takes ~2.5 h on 4 CPU cores.

## Limits
Scalar, thin mask, constant-threshold resist, 2D, single source, toy clips.
Numbers (nm, counts) are specific to this toy; the *mechanisms* are what the
experiments test.  See REPORT.md.
