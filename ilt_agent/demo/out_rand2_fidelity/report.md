# ILT clip analysis

Extent 596 x 596 nm, pixel 2.328 nm, 27 process conditions (nominal: `d1.00_f+0_m+0`).

Spec: {"w_min_nm": 28.0, "s_min_nm": 28.0, "margin_nm": 2.0, "epe_tol_nm": 5.0, "mrc_width_nm": 10.0, "mrc_space_nm": 10.0, "mrc_area_nm2": 150.0, "merge_nm": 12.0, "border_nm": 30.0, "corner_nm": 12.0}

## 1. Global

| layer | count | area_total_nm2 | area_min_nm2 | vertices_median | compactness_median |
|---|---|---|---|---|---|
| target | 20 | 103227.06176757812 | 552.85693359375 | 164.0 | 2.0599702977232353 |
| mask | 25 | 149293.052734375 | 70.462158203125 | 183.0 | 2.365237841436581 |


Detached assist features on mask: 6; mask area outside target: 51166 nm2; printed/target area at nominal: 0.977.


![overview](overview.png)

## 2. Printability (all conditions)

Counts: `{"bridge": {"catastrophic": 7}, "space": {"violation": 9, "marginal": 10}, "pinch": {"violation": 44, "marginal": 31}, "epe": {"info": 30}}`; conditions with most findings: [('d1.00_f+0_m+0', 23), ('d0.95_f+40_m-1', 20), ('d0.95_f-40_m-1', 20), ('d0.95_f+0_m+1', 19), ('d0.95_f+40_m+0', 19)]


| id | type | level | x_nm | y_nm | value | n_conditions |
|---|---|---|---|---|---|---|
| F000 | bridge | catastrophic | 557.6 | 345.7 | 0.0 | 16 |
| F001 | bridge | catastrophic | 403.9 | 471.4 | 0.0 | 15 |
| F002 | bridge | catastrophic | 331.8 | 322.4 | 0.0 | 12 |
| F003 | bridge | catastrophic | 33.8 | 478.4 | 0.0 | 9 |
| F004 | bridge | catastrophic | 33.8 | 80.3 | 0.0 | 1 |
| F005 | bridge | catastrophic | 257.3 | 313.1 | 0.0 | 8 |
| F006 | bridge | catastrophic | 36.1 | 278.2 | 0.0 | 5 |
| F007 | space | violation | 36.1 | 80.3 | 4.66 | 3 |
| F008 | pinch | violation | 261.9 | 66.4 | 6.58 | 1 |
| F009 | pinch | violation | 331.8 | 285.2 | 9.31 | 10 |
| F010 | pinch | violation | 259.6 | 143.2 | 9.31 | 17 |
| F011 | pinch | violation | 557.6 | 129.2 | 9.31 | 6 |
| F012 | space | violation | 261.9 | 306.1 | 9.31 | 5 |
| F013 | space | violation | 408.6 | 464.5 | 13.97 | 4 |
| F014 | space | violation | 331.8 | 324.8 | 13.97 | 3 |
| F015 | space | violation | 559.9 | 352.7 | 13.97 | 3 |
| F016 | space | violation | 483.1 | 403.9 | 13.97 | 4 |
| F017 | pinch | violation | 410.9 | 61.7 | 14.72 | 1 |
| F018 | pinch | violation | 408.6 | 539.0 | 18.62 | 3 |
| F019 | pinch | violation | 485.4 | 552.9 | 18.62 | 9 |
| F020 | pinch | violation | 557.6 | 406.3 | 18.62 | 7 |
| F021 | pinch | violation | 410.9 | 515.7 | 18.62 | 5 |
| F022 | pinch | violation | 38.4 | 187.4 | 18.62 | 2 |
| F023 | pinch | violation | 410.9 | 566.9 | 18.62 | 1 |
| F024 | pinch | violation | 410.9 | 431.9 | 18.62 | 4 |


## 3. EPE and PV band

Nominal EPE: {'mean': 0.34, 'rms': 2.38, 'p99_abs': 7.56, 'frac_above_tol': 0.0433}; PV-band width along edges: {'median': 6.98, 'p95': 15.33, 'max': 58.94}; PV-band area / target area: 0.672.


| x_nm | y_nm | value_nm |
|---|---|---|
| 356.2 | 367.1 | -9.52 |
| 239.8 | 359.2 | -9.31 |
| 186.0 | 470.3 | 9.03 |
| 188.6 | 305.0 | 8.68 |
| 356.2 | 371.1 | -8.33 |
| 256.5 | 519.2 | 8.09 |
| 109.2 | 37.2 | 7.97 |
| 557.5 | 109.4 | 7.94 |


![pv band](overview_pvband.png)


![ils](overview_ils.png)

## 4. Mask rules

Rules {'min_width_nm': 10.0, 'min_space_nm': 10.0, 'min_area_nm2': 150.0}: width violations 81, space violations 16, small islands 1, sharp corners (r<4 nm) 31.

## 5. Regional view and anomalies

Tile 149.0 nm, 16 tiles. Feature medians: `{"target_density": 0.3704, "mask_density": 0.4978, "mask_target_xor": 0.1482, "epe_rms_nom": 3.5395, "epe_maxabs_nom": 9.7108, "pv_band_frac": 0.2179, "n_findings": 8.0, "mask_perimeter_per_area": 0.0581, "min_edge_ils": 0.0163}`


| x0_nm | y0_nm | x1_nm | y1_nm | score | n_findings | driving_features |
|---|---|---|---|---|---|---|
| 447.0 | 0.0 | 596.0 | 149.0 | 0.792 | 3.0 | ['pv_band_frac (|z|=3.5)', 'mask_perimeter_per_area (|z|=2.7)', 'target_density (|z|=2.5)'] |
| 0.0 | 298.0 | 149.0 | 447.0 | 0.728 | 8.0 | ['min_edge_ils (|z|=5.1)', 'mask_target_xor (|z|=1.9)', 'target_density (|z|=0.9)'] |
| 0.0 | 447.0 | 149.0 | 596.0 | 0.69 | 10.0 | ['epe_maxabs_nom (|z|=5.8)', 'epe_rms_nom (|z|=1.4)', 'pv_band_frac (|z|=1.1)'] |
| 298.0 | 447.0 | 447.0 | 596.0 | 0.679 | 7.0 | ['epe_maxabs_nom (|z|=6.0)', 'pv_band_frac (|z|=1.4)', 'min_edge_ils (|z|=0.7)'] |
| 298.0 | 298.0 | 447.0 | 447.0 | 0.497 | 15.0 | ['epe_maxabs_nom (|z|=3.4)', 'n_findings (|z|=1.9)', 'mask_target_xor (|z|=1.0)'] |


## 6. Hotspot pattern classes

| cluster | size | types | representative | members |
|---|---|---|---|---|
| 2 | 2 | {'bridge': 1, 'space': 1} | F002 | F002, F014 |
| 49 | 2 | {'pinch': 2} | F050 | F050, F057 |
| 0 | 1 | {'bridge': 1} | F000 | F000 |
| 1 | 1 | {'bridge': 1} | F001 | F001 |
| 3 | 1 | {'bridge': 1} | F003 | F003 |
| 4 | 1 | {'bridge': 1} | F004 | F004 |
| 5 | 1 | {'bridge': 1} | F005 | F005 |
| 6 | 1 | {'bridge': 1} | F006 | F006 |
| 7 | 1 | {'space': 1} | F007 | F007 |
| 8 | 1 | {'pinch': 1} | F008 | F008 |
| 9 | 1 | {'pinch': 1} | F009 | F009 |
| 10 | 1 | {'pinch': 1} | F010 | F010 |
| 11 | 1 | {'pinch': 1} | F011 | F011 |
| 12 | 1 | {'space': 1} | F012 | F012 |
| 13 | 1 | {'space': 1} | F013 | F013 |
| 14 | 1 | {'space': 1} | F015 | F015 |
| 15 | 1 | {'space': 1} | F016 | F016 |
| 16 | 1 | {'pinch': 1} | F017 | F017 |
| 17 | 1 | {'pinch': 1} | F018 | F018 |
| 18 | 1 | {'pinch': 1} | F019 | F019 |
| 19 | 1 | {'pinch': 1} | F020 | F020 |
| 20 | 1 | {'pinch': 1} | F021 | F021 |
| 21 | 1 | {'pinch': 1} | F022 | F022 |
| 22 | 1 | {'pinch': 1} | F023 | F023 |
| 23 | 1 | {'pinch': 1} | F024 | F024 |
| 24 | 1 | {'pinch': 1} | F025 | F025 |
| 25 | 1 | {'pinch': 1} | F026 | F026 |
| 26 | 1 | {'pinch': 1} | F027 | F027 |
| 27 | 1 | {'pinch': 1} | F028 | F028 |
| 28 | 1 | {'pinch': 1} | F029 | F029 |
| 29 | 1 | {'pinch': 1} | F030 | F030 |
| 30 | 1 | {'pinch': 1} | F031 | F031 |
| 31 | 1 | {'pinch': 1} | F032 | F032 |
| 32 | 1 | {'pinch': 1} | F033 | F033 |
| 33 | 1 | {'pinch': 1} | F034 | F034 |
| 34 | 1 | {'pinch': 1} | F035 | F035 |
| 35 | 1 | {'pinch': 1} | F036 | F036 |
| 36 | 1 | {'pinch': 1} | F037 | F037 |
| 37 | 1 | {'pinch': 1} | F038 | F038 |
| 38 | 1 | {'pinch': 1} | F039 | F039 |
| 39 | 1 | {'pinch': 1} | F040 | F040 |
| 40 | 1 | {'pinch': 1} | F041 | F041 |
| 41 | 1 | {'pinch': 1} | F042 | F042 |
| 42 | 1 | {'pinch': 1} | F043 | F043 |
| 43 | 1 | {'pinch': 1} | F044 | F044 |
| 44 | 1 | {'pinch': 1} | F045 | F045 |
| 45 | 1 | {'pinch': 1} | F046 | F046 |
| 46 | 1 | {'space': 1} | F047 | F047 |
| 47 | 1 | {'space': 1} | F048 | F048 |
| 48 | 1 | {'space': 1} | F049 | F049 |
| 50 | 1 | {'pinch': 1} | F051 | F051 |
| 51 | 1 | {'pinch': 1} | F052 | F052 |
| 52 | 1 | {'pinch': 1} | F053 | F053 |
| 53 | 1 | {'pinch': 1} | F054 | F054 |
| 54 | 1 | {'pinch': 1} | F055 | F055 |
| 55 | 1 | {'pinch': 1} | F056 | F056 |
| 56 | 1 | {'pinch': 1} | F058 | F058 |
| 57 | 1 | {'pinch': 1} | F059 | F059 |


## 7. Close-ups

### F000 bridge at (557.6, 345.7) nm

worst width 4.7 nm, worst space 14.0 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.143; edge ILS {'p5': 0.0146, 'median': 0.0398}

![](finding_F000.png)

### F001 bridge at (403.9, 471.4) nm

worst width 4.7 nm, worst space 14.0 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.13; edge ILS {'p5': 0.0181, 'median': 0.0424}

![](finding_F001.png)

### F002 bridge at (331.8, 322.4) nm

worst width 4.7 nm, worst space 14.0 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.147; edge ILS {'p5': 0.0114, 'median': 0.0378}

![](finding_F002.png)

### F003 bridge at (33.8, 478.4) nm

worst width 4.7 nm, worst space 27.9 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.134; edge ILS {'p5': 0.0159, 'median': 0.0416}

![](finding_F003.png)

### anomaly pv_band_frac (|z|=3.5), mask_perimeter_per_area (|z|=2.7) at (521.5, 74.5) nm

worst width 9.3 nm, worst space 32.6 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.197; edge ILS {'p5': 0.0121, 'median': 0.0311}

![](anomaly_521_74.png)

### anomaly min_edge_ils (|z|=5.1), mask_target_xor (|z|=1.9) at (74.5, 372.5) nm

worst width 23.3 nm, worst space 4.7 nm (spec {'w_min_nm': 28.0, 's_min_nm': 28.0}); mask outside target 0.083; edge ILS {'p5': 0.0397, 'median': 0.0419}

![](anomaly_74_372.png)


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


Mask: 2 detached assists, 1 mask shapes joining different targets:

| targets | x_nm | y_nm | neck_width_nm | target_gap_nm |
|---|---|---|---|---|
| [2, 5] | 147.8 | 24.4 | 21.0 | 37.2 |


Mask bias along target normals: {'min': -14.0, 'p5': -4.5, 'median': 2.3, 'p95': 11.6, 'max': 19.3}


Gauges ({'tip-to-tip': 4, 'width': 34, 'space': 27}), worst margin first:

| id | kind | ref_nm | design_nm | nominal_cd | min_cd | worst_condition | margin_to_spec_nm | dCD_per_1pct_dose_nm | MEEF |
|---|---|---|---|---|---|---|---|---|---|
| G035 | tip-to-tip | [37.2, 279.8] | 60.0 | 51.22 | 0.0 | d1.00_f-40_m+1 | -28.0 | -2.328 | 11.641 |
| G037 | tip-to-tip | [37.2, 478.2] | 50.0 | 27.94 | 0.0 | d1.00_f-40_m+0 | -28.0 | -4.656 | 14.842 |
| G000 | width | [37.2, 161.4] | 30.0 | 30.27 | 6.43 | d0.95_f+40_m-1 | -21.57 | 0.493 | 2.91 |
| G059 | tip-to-tip | [484.2, 399.5] | 60.0 | 51.22 | 12.82 | d1.05_f-40_m+1 | -15.18 | -1.397 | 6.984 |
| G012 | width | [260.8, 132.7] | 30.0 | 27.94 | 13.97 | d0.95_f-40_m-1 | -14.03 | 0.931 | 2.91 |
| G024 | width | [409.8, 30.5] | 30.0 | 32.59 | 13.97 | d0.95_f+40_m-1 | -14.03 | 0.466 | 3.492 |
| G025 | width | [409.8, 31.6] | 30.0 | 32.49 | 13.97 | d0.95_f+40_m-1 | -14.03 | 0.466 | 3.466 |
| G022 | width | [409.8, 540.5] | 30.0 | 27.94 | 18.63 | d0.95_f+0_m-1 | -9.37 | 0.862 | 2.328 |


Line-end pull-back (worst first):

| poly | x_nm | y_nm | facing | gap_nm | max_pullback_nm | min_pullback_nm | worst_condition |
|---|---|---|---|---|---|---|---|
| 0 | 37.2 | 112.0 | -y | 60.0 | 40.0 | -30.0 | d0.95_f+40_m-1 |
| 0 | 37.2 | 249.8 | +y | 60.0 | 18.22 | -30.0 | d0.95_f+40_m-1 |
| 15 | 409.8 | 62.1 | +y | 140.0 | 15.54 | -13.63 | d0.95_f+40_m-1 |
| 19 | 558.8 | 372.4 | -y | 50.0 | 14.07 | -25.0 | d0.95_f+0_m-1 |
| 12 | 335.2 | 96.9 | -y | 80.0 | 13.76 | -15.42 | d0.95_f+40_m-1 |
| 19 | 558.8 | 564.0 | +y | None | 13.47 | -23.78 | d0.95_f+0_m-1 |
| 1 | 37.2 | 453.2 | +y | 50.0 | 13.18 | -25.0 | d0.95_f+0_m-1 |
| 14 | 409.8 | 502.0 | -y | 60.0 | 10.19 | -30.0 | d0.95_f+0_m-1 |
