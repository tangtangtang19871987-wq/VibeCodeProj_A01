# ILT clip analysis

Extent 596 x 596 nm, pixel 2.328 nm, 27 process conditions (nominal: `d1.00_f+0_m+0`).

Spec: {"w_min_nm": 28.0, "s_min_nm": 28.0, "margin_nm": 2.0, "epe_tol_nm": 5.0, "mrc_width_nm": 10.0, "mrc_space_nm": 10.0, "mrc_area_nm2": 150.0, "merge_nm": 12.0, "border_nm": 30.0, "corner_nm": 12.0}

## 1. Global

| layer | count | area_total_nm2 | area_min_nm2 | vertices_median | compactness_median |
|---|---|---|---|---|---|
| target | 20 | 103227.06176757812 | 552.85693359375 | 164.0 | 2.0599702977232353 |
| mask | 17 | 166575.2520751953 | 2184.326904296875 | 212.0 | 2.420912828749849 |


Detached assist features on mask: 1; mask area outside target: 65237 nm2; printed/target area at nominal: 1.303.


![overview](overview.png)

## 2. Printability (all conditions)

Counts: `{"bridge": {"catastrophic": 2}, "space": {"violation": 2, "marginal": 32}, "pinch": {"marginal": 12}, "epe": {"info": 61}}`; conditions with most findings: [('d1.05_f+40_m+1', 22), ('d1.05_f+0_m+1', 20), ('d1.05_f-40_m+1', 20), ('d0.95_f-40_m-1', 6), ('d1.00_f+40_m+1', 6)]


| id | type | level | x_nm | y_nm | value | n_conditions |
|---|---|---|---|---|---|---|
| F000 | bridge | catastrophic | 448.2 | 422.6 | 0.0 | 22 |
| F001 | bridge | catastrophic | 78.0 | 78.0 | 0.0 | 17 |
| F002 | space | violation | 450.5 | 438.9 | 6.58 | 5 |
| F003 | space | violation | 336.4 | 64.0 | 23.28 | 1 |
| F004 | space | marginal | 220.0 | 271.2 | 26.34 | 5 |
| F005 | space | marginal | 525.0 | 359.7 | 26.34 | 4 |
| F006 | pinch | marginal | 264.2 | 359.7 | 27.15 | 11 |
| F007 | pinch | marginal | 250.3 | 254.9 | 27.15 | 1 |
| F008 | space | marginal | 376.0 | 511.0 | 27.15 | 3 |
| F009 | space | marginal | 294.5 | 506.4 | 27.15 | 3 |
| F010 | space | marginal | 450.5 | 348.1 | 27.15 | 1 |
| F011 | pinch | marginal | 110.6 | 380.6 | 27.94 | 2 |
| F012 | pinch | marginal | 487.7 | 564.6 | 27.94 | 2 |
| F013 | space | marginal | 443.5 | 324.8 | 27.94 | 4 |
| F014 | space | marginal | 150.2 | 310.8 | 27.94 | 4 |
| F015 | space | marginal | 443.5 | 315.5 | 27.94 | 2 |
| F016 | space | marginal | 378.3 | 390.0 | 27.94 | 6 |
| F017 | space | marginal | 222.3 | 403.9 | 27.94 | 3 |
| F018 | space | marginal | 299.2 | 422.6 | 27.94 | 1 |
| F019 | space | marginal | 220.0 | 427.2 | 27.94 | 3 |
| F020 | space | marginal | 371.3 | 543.6 | 27.94 | 2 |
| F021 | space | marginal | 80.3 | 552.9 | 27.94 | 1 |
| F022 | space | marginal | 520.3 | 480.8 | 27.94 | 3 |
| F023 | space | marginal | 73.3 | 357.4 | 27.94 | 3 |
| F024 | pinch | marginal | 336.4 | 364.4 | 27.94 | 2 |


## 3. EPE and PV band

Nominal EPE: {'mean': 3.4, 'rms': 8.12, 'p99_abs': 26.49, 'frac_above_tol': 0.3908}; PV-band width along edges: {'median': 6.54, 'p95': 11.39, 'max': 23.99}; PV-band area / target area: 0.578.


| x_nm | y_nm | value_nm |
|---|---|---|
| 51.2 | 153.7 | 30.0 |
| 51.2 | 157.7 | 29.99 |
| 51.2 | 149.7 | 29.42 |
| 51.2 | 161.7 | 28.79 |
| 51.2 | 145.7 | 28.61 |
| 51.2 | 165.7 | 28.12 |
| 51.2 | 169.7 | 27.94 |
| 51.2 | 173.7 | 27.94 |


![pv band](overview_pvband.png)


![ils](overview_ils.png)

## 4. Mask rules

Rules {'min_width_nm': 10.0, 'min_space_nm': 10.0, 'min_area_nm2': 150.0}: width violations 53, space violations 11, small islands 0, sharp corners (r<4 nm) 11.

## 5. Regional view and anomalies

Tile 149.0 nm, 16 tiles. Feature medians: `{"target_density": 0.3704, "mask_density": 0.5076, "mask_target_xor": 0.1681, "epe_rms_nom": 8.8187, "epe_maxabs_nom": 21.6327, "pv_band_frac": 0.1785, "n_findings": 6.0, "mask_perimeter_per_area": 0.0502, "min_edge_ils": 0.0166}`


| x0_nm | y0_nm | x1_nm | y1_nm | score | n_findings | driving_features |
|---|---|---|---|---|---|---|
| 298.0 | 0.0 | 447.0 | 149.0 | 0.805 | 4.0 | ['mask_perimeter_per_area (|z|=5.5)', 'pv_band_frac (|z|=1.7)', 'target_density (|z|=1.4)'] |
| 447.0 | 0.0 | 596.0 | 149.0 | 0.76 | 1.0 | ['mask_target_xor (|z|=3.1)', 'target_density (|z|=2.5)', 'pv_band_frac (|z|=2.3)'] |
| 149.0 | 447.0 | 298.0 | 596.0 | 0.645 | 7.0 | ['mask_perimeter_per_area (|z|=3.9)', 'mask_density (|z|=3.7)', 'target_density (|z|=1.6)'] |
| 0.0 | 149.0 | 149.0 | 298.0 | 0.465 | 4.0 | ['mask_density (|z|=1.7)', 'pv_band_frac (|z|=1.7)', 'epe_rms_nom (|z|=1.6)'] |
| 298.0 | 298.0 | 447.0 | 447.0 | 0.436 | 14.0 | ['n_findings (|z|=2.7)', 'mask_perimeter_per_area (|z|=2.4)', 'min_edge_ils (|z|=0.7)'] |


## 6. Hotspot pattern classes

| cluster | size | types | representative | members |
|---|---|---|---|---|
| 0 | 1 | {'bridge': 1} | F000 | F000 |
| 1 | 1 | {'bridge': 1} | F001 | F001 |
| 2 | 1 | {'space': 1} | F002 | F002 |
| 3 | 1 | {'space': 1} | F003 | F003 |


## 7. Close-ups

### F000 bridge at (448.2, 422.6) nm

worst width 4.7 nm, worst space 6.6 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.186; edge ILS {'p5': 0.0114, 'median': 0.0303}

![](finding_F000.png)

### F001 bridge at (78.0, 78.0) nm

worst width 4.7 nm, worst space 4.7 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.357; edge ILS {'p5': 0.0166, 'median': 0.0361}

![](finding_F001.png)

### F002 space at (450.5, 438.9) nm

worst width 4.7 nm, worst space 6.6 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.207; edge ILS {'p5': 0.0114, 'median': 0.0302}

![](finding_F002.png)

### F003 space at (336.4, 64.0) nm

worst width 27.9 nm, worst space 23.3 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.23; edge ILS {'p5': 0.0163, 'median': 0.0264}

![](finding_F003.png)

### anomaly mask_perimeter_per_area (|z|=5.5), pv_band_frac (|z|=1.7) at (372.5, 74.5) nm

worst width 28.3 nm, worst space 16.8 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.234; edge ILS {'p5': 0.0142, 'median': 0.0303}

![](anomaly_372_74.png)

### anomaly mask_target_xor (|z|=3.1), target_density (|z|=2.5) at (521.5, 74.5) nm

worst width 10.4 nm, worst space 13.2 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.424; edge ILS {'p5': 0.0074, 'median': 0.033}

![](anomaly_521_74.png)


## 8. Layout reading (geometry toolkit)

Orientation V (purity 0.805), pitch 74.51 nm, tracks [0, 1, 2, 3, 4, 5, 6, 7], line widths [30.0, 37.0, 44.0] nm, 19 line ends ({'end-side': 7, 'end-end': 8, 'none': 4}), polygon classes {'line': 18, 'island': 2}; source: target_polygons from meta.


| measure | n | min | p5 | median | max |
|---|---|---|---|---|---|
| width_side_edges | 44 | 16.9 | 30.0 | 37.0 | 52.0 |
| line_end_widths | 26 | 16.9 | 20.1 | 37.0 | 44.0 |
| space_side-side | 30 | 30.5 | 30.5 | 37.5 | 112.0 |
| space_end-end | 8 | 50.0 | 50.0 | 60.0 | 140.0 |
| space_end-side | 17 | 37.5 | 37.5 | 60.0 | 183.0 |
| space_notch | 0 |  |  |  |  |


Tightest line ends:

| poly | x_nm | y_nm | facing | width_nm | gap_nm | gap_kind | context |
|---|---|---|---|---|---|---|---|
| 1 | 37.2 | 453.2 | +y | 37.0 | 50.0 | end-end | semi-isolated |
| 2 | 37.2 | 503.2 | -y | 37.0 | 50.0 | end-end | semi-isolated |
| 19 | 558.8 | 372.4 | -y | 30.0 | 50.0 | end-side | isolated |
| 0 | 37.2 | 112.0 | -y | 30.0 | 60.0 | end-side | isolated |
| 0 | 37.2 | 249.8 | +y | 30.0 | 60.0 | end-end | isolated |
| 1 | 37.2 | 309.8 | -y | 37.0 | 60.0 | end-end | semi-isolated |
| 9 | 260.8 | 344.0 | -y | 44.0 | 60.0 | end-side | semi-isolated |
| 12 | 335.2 | 293.7 | +y | 37.0 | 60.0 | end-side | semi-isolated |
| 14 | 409.8 | 502.0 | -y | 30.0 | 60.0 | end-side | dense |
| 16 | 484.2 | 369.5 | +y | 44.0 | 60.0 | end-end | semi-isolated |


Mask: 0 detached assists, 3 mask shapes joining different targets:

| targets | x_nm | y_nm | neck_width_nm | target_gap_nm |
|---|---|---|---|---|
| [2, 9] | 82.6 | 80.3 | 53.5 | 91.8 |
| [3, 7] | 334.1 | 57.0 | 17.4 | 83.8 |
| [4, 10, 18] | 429.5 | 131.5 | 30.6 | 142.4 |


Mask bias along target normals: {'min': -0.0, 'p5': 0.0, 'median': 2.3, 'p95': 18.7, 'max': 27.6}


Gauges ({'space': 27, 'width': 34, 'tip-to-tip': 4}), worst margin first:

| id | kind | ref_nm | design_nm | nominal_cd | min_cd | worst_condition | margin_to_spec_nm | dCD_per_1pct_dose_nm | MEEF |
|---|---|---|---|---|---|---|---|---|---|
| G045 | space | [110.0, 89.4] | 108.5 | 0.0 | 0.0 | d0.95_f+0_m-1 | -28.0 | 0.0 | -0.0 |
| G047 | space | [227.0, 173.2] | 37.5 | 32.12 | 0.0 | d1.05_f+0_m+1 | -28.0 | -0.233 | 0.582 |
| G048 | space | [298.0, 431.2] | 30.5 | 32.59 | 27.94 | d1.05_f+0_m+1 | -0.06 | 0.0 | 1.594 |
| G049 | space | [223.5, 431.2] | 30.5 | 32.59 | 27.94 | d1.05_f+0_m+1 | -0.06 | -0.466 | 1.164 |
| G054 | space | [445.2, 322.0] | 34.0 | 34.61 | 29.95 | d1.05_f-40_m+1 | 1.95 | -0.264 | 1.746 |
| G007 | width | [111.8, 495.0] | 37.0 | 41.91 | 30.27 | d0.95_f+40_m-1 | 2.27 | 0.445 | 1.216 |
| G020 | width | [409.8, 277.1] | 37.0 | 38.45 | 30.27 | d0.95_f-40_m-1 | 2.27 | 0.466 | 1.164 |
| G025 | width | [409.8, 31.6] | 30.0 | 39.58 | 30.27 | d0.95_f+40_m-1 | 2.27 | 0.243 | 2.301 |


Line-end pull-back (worst first):

| poly | x_nm | y_nm | facing | gap_nm | max_pullback_nm | min_pullback_nm | worst_condition |
|---|---|---|---|---|---|---|---|
| 19 | 558.8 | 372.4 | -y | 50.0 | 17.59 | -7.98 | d0.95_f+40_m-1 |
| 1 | 37.2 | 453.2 | +y | 50.0 | 15.51 | -7.77 | d0.95_f+0_m-1 |
| 12 | 335.2 | 293.7 | +y | 60.0 | 15.42 | -12.52 | d0.95_f+0_m-1 |
| 2 | 37.2 | 503.2 | -y | 50.0 | 14.88 | -7.31 | d0.95_f+0_m-1 |
| 14 | 409.8 | 502.0 | -y | 60.0 | 14.84 | -15.42 | d0.95_f+0_m-1 |
| 19 | 558.8 | 564.0 | +y | None | 14.56 | -18.03 | d0.95_f+0_m-1 |
| 16 | 484.2 | 369.5 | +y | 60.0 | 14.53 | 1.66 | d0.95_f+0_m-1 |
| 17 | 484.2 | 429.5 | -y | 60.0 | 13.94 | -0.03 | d0.95_f+0_m-1 |
