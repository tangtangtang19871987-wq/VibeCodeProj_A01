# Model Order Reduction in Computational Electromagnetics and Nanophotonic Inverse Design (2010–2026): Physics-Based Methods Transferable to Computational Lithography

Scope note: physics-based (projection, interpolation, modal, low-rank operator) reduction only. ML surrogates (deep surrogates, PINNs, neural operators, e.g. Pestourie's active-learning deep surrogates) are out of scope. They are mentioned only where they draw a boundary with the physics-based methods.

Verification legend: entries marked **[V]** had title, authors, venue, and DOI/URL confirmed by web search in this session. Entries marked **[B]** are bibliographic data from background knowledge that were *not* re-verified in this session (arXiv full-text fetches failed because of DNS errors). The report writer should spot-check [B] DOIs before final publication.

---

## Q1. Reduced basis methods (RBM) for parametrized Maxwell problems: certified RB, error estimators, greedy sampling, offline/online split

### Takeaway
Certified RBM for time-harmonic Maxwell was mature by about 2010–2015 (DG, FEM, and BEM/EFIE truth solvers; residual-based a-posteriori bounds; greedy sampling). A lithography-specific RBM already exists: Pomplun, Zschiedrich, Burger and Schmidt (ZIB/JCMwave, with Nikon) applied it to rigorous 3D mask transmission for OPC (2009) and SMO (2010). It is the most directly transferable prior art. Its known weak points are the inf-sup constant (near resonances) and the cost of affine decomposition for geometry parameters.

### Cited Findings
**Foundational and review**
- Hesthaven, Rozza & Stamm, *Certified Reduced Basis Methods for Parametrized Partial Differential Equations*, SpringerBriefs in Mathematics (2016). This is the standard reference for the offline/online split, residual-based a-posteriori estimators (dual norm of the residual divided by a lower bound of the coercivity/inf-sup constant), the Successive Constraint Method, and weak-greedy sampling. **[B]** — [doi:10.1007/978-3-319-22470-1](https://doi.org/10.1007/978-3-319-22470-1)
- Benner, Gugercin & Willcox, "A survey of projection-based model reduction methods for parametric dynamical systems," *SIAM Review* 57(4):483–531 (2015). It unifies RB/POD, Krylov/moment matching, and balanced truncation in a parametric setting, and covers how to build global versus interpolated local bases. **[B]** — [doi:10.1137/130932715](https://doi.org/10.1137/130932715)

**Certified RB for harmonic Maxwell**
- Chen, Hesthaven, Maday & Rodríguez, "Certified reduced basis methods and output bounds for the harmonic Maxwell's equations," *SIAM J. Sci. Comput.* 32(2):970–996 (2010). **[V]**
  - The truth solver is a discontinuous Galerkin discretization, and the reduced basis is built by greedy sampling.
  - Separate primal and dual RB spaces give certified *output* bounds and lower the offline cost.
  - Tests cover resonant behavior and material variation.
  - [doi:10.1137/09075250X](https://doi.org/10.1137/09075250X); [EPFL record](https://infoscience.epfl.ch/record/190379)
- Hess & Benner, "Fast evaluation of time-harmonic Maxwell's equations using the reduced basis method," *IEEE Trans. Microw. Theory Techn.* 61(6):2265–2274 (2013). **[V]**
  - Rigorous error estimators cover variation of material and geometry parameters.
  - The model order is reduced by a **factor of 50–100**, and compute time by a **factor of about 200 or more**, for microwave circuit-element examples.
  - [doi:10.1109/TMTT.2013.2258167](https://doi.org/10.1109/TMTT.2013.2258167); [MPI preprint](https://csc.mpi-magdeburg.mpg.de/preprints/2012/17/)
- Follow-up: Hess & Benner, "Output error estimates in RBM for time-harmonic Maxwell" (ENUMATH 2015, published 2016). It uses an expanded formulation so that the quadratic impedance output becomes compliant. — [summary via search record](https://csc.mpi-magdeburg.mpg.de/preprints/2012/17/) **[V, abstract-level]**
- Hess, Grundel & Benner, "Estimating the inf-sup constant in reduced basis methods for time-harmonic Maxwell's equations," *IEEE TMTT* 63(11) (2015). Inf-sup estimation is the bottleneck for certification near resonances. **[B]**
- "Inf-Sup-Constant-Free State Error Estimator for Model Order Reduction of Parametric Systems in Electromagnetics" (Chellappa, Feng, de la Rubia & Benner, IEEE TMTT 2023 — authors and venue are **[B]**). It removes the need for the inf-sup lower bound, which is the classic weak point for wave problems. — [arXiv:2104.12802](https://arxiv.org/abs/2104.12802) **[V title]**
- De la Rubia, Fernández & Salazar-Palma, "Reliable fast frequency sweep for microwave devices via the reduced-basis method," *IEEE TMTT* 57(12) (2009). It is an RB frequency sweep with a FEM truth solver and an error indicator. **[B]**
- "Fast Plasma Frequency Sweep in Drude-like EM Scatterers via the Reduced-Basis Method" (2024) is a recent RBM for dispersive-material parameter sweeps. — [arXiv:2410.11004](https://arxiv.org/abs/2410.11004) **[V title only; authors not verified]**

**RB for integral-equation (BEM/EFIE) scattering**
- Fares, Hesthaven, Maday & Stamm, "The reduced basis method for the electric field integral equation," *J. Comput. Phys.* 230(14):5532–5555 (2011). **[V]**
  - The truth solver is a BEM/EFIE discretization.
  - Parameters are wavenumber, incidence angle, and polarization.
  - Online outputs (for example, RCS) are cheap.
  - EIM handles the non-affine dependence of the Green's kernel on wavenumber and angle.
  - [doi:10.1016/j.jcp.2011.03.023](https://doi.org/10.1016/j.jcp.2011.03.023); [PDF](https://www.dam.brown.edu/AFOSR_MURI_pub/JCPRBMIE_hesthaven_reduced.pdf)
- Hesthaven, Stamm & Zhang, "Certified reduced basis method for the electric field integral equation," *SIAM J. Sci. Comput.* (2012). It replaces the simplified estimator of Fares et al. with a rigorous a-posteriori bound. — [EPFL record](https://infoscience.epfl.ch/record/190445?ln=en) **[V title/venue; volume/pages B]**
- Hesthaven, Stamm & Zhang, "Efficient greedy algorithms for high-dimensional parameter spaces with application to empirical interpolation and reduced basis methods," *ESAIM: M2AN* 48(1):259–283 (2014). It uses a saturation assumption and adaptive training-set enrichment to make greedy sampling tractable in many parameter dimensions. This matters for SMO/ILT-like parameter counts. **[B]**

**RB for lithography masks and scatterometry (ZIB / JCMwave)**
- Pomplun, Zschiedrich, Burger, Schmidt, Tyminski, Flagello & Toshiharu, "Reduced basis method for source mask optimization," *Proc. SPIE* 7823, Photomask Technology 2010, 78230E. **[V]**
  - The truth solver is offline FEM (JCMsuite).
  - The reduced model is then solved online "very fast for arbitrary illumination and geometrical parameters" (for example, OPC feature dimensions and line widths).
  - Rigorous error estimators certify the RB solutions and drive self-adaptive basis construction.
  - The authors claim the method "dramatically reduces computational costs of the optimization procedure while providing accuracy superior to the approaches involving simplified mask models." A 3D SMO example is shown.
  - [doi:10.1117/12.866101](https://doi.org/10.1117/12.866101); [arXiv:1011.2902](https://arxiv.org/abs/1011.2902)
- Pomplun, Zschiedrich, Burger & Schmidt, "Reduced basis method for computational lithography" (SPIE 2009 proceedings; arXiv 2010). It applies RBM to OPC of a contact-hole array. — [arXiv:1011.2654](https://arxiv.org/abs/1011.2654) **[V title/arXiv; SPIE volume B]**
- Pomplun & Schmidt, "Accelerated a posteriori error estimation for the reduced basis method with application to 3D electromagnetic scattering problems," *SIAM J. Sci. Comput.* 32(2):498–520 (2010). It speeds up evaluation of the residual dual norm for 3D vector FEM, whose affine expansion has many terms. **[B — search did not return this record; verify]**
- Hammerschmidt, Herrmann, Pomplun, Zschiedrich, Burger & Schmidt, "Reduced basis method for Maxwell's equations with resonance phenomena," *Proc. SPIE* 9630 (2015). It is RBM for resonant nano-optical structures and scatterometry. **[B]**

### Inferences
- ZIB/JCMwave showed in 2009–2010 that certified RBM fits the rigorous-mask-3D part of computational lithography. The mask near field is treated as a parametric Maxwell solve over (incidence angle, polarization, feature CDs). The set of source points in a partially coherent source is a natural many-query parameter set: one mask with many plane-wave incidences per source pixel. This is the classic RB "many right-hand sides / parameter sweep" regime.
- What is missing for ILT: RBM handles low-dimensional *shape* parameters (a few CDs) well. Pixel-based ILT has 10^4–10^6 design variables, so a global RB over the mask design space is infeasible. Transfer is most plausible either (a) per local mask tile or library of patterns (see the LPA/overlapping domains in Q6), or (b) over illumination parameters only.
- The intensity |E|² is a *quadratic* output, so its RB output evaluation is exactly affine: a precomputed N×N Hermitian matrix per output point/pupil. It does not need EIM. Only the resist sigmoid and threshold need empirical interpolation (see Q3).

### Gaps
- No published speedup numbers or RB dimensions for the 2010 SMO paper could be extracted (arXiv full text unreachable in this session).
- I found no follow-up from the ZIB group after about 2011 that applies RBM to full-chip or ILT-scale lithography. The search showed no such work, which suggests the line was not pursued at production scale; the reasons are undocumented.
- The SIAM J. Sci. Comput. citation for Pomplun & Schmidt (2010) could not be verified.

---

## Q2. Krylov / moment-matching MOR (PRIMA, multipoint Padé, parametric Krylov) for frequency and parameter sweeps

### Takeaway
Krylov moment matching (PVL → PRIMA → multiparameter moment matching) is the standard tool for fast frequency sweeps of FEM/PEEC/IE models in microwave and interconnect engineering. Rational-interpolation methods (vector fitting, AAA, Loewner) are its data-driven counterpart. For lithography the "frequency" axis is mostly fixed (one wavelength, or a narrow EUV band). The useful analogues are sweeps over incidence angle and source point, defocus, and mask-bias parameters, where multipoint or parametric Krylov and RB bases are interchangeable.

### Cited Findings
- **Foundations (pre-2010):**
  - Feldmann & Freund, "Efficient linear circuit analysis by Padé approximation via the Lanczos process" (PVL), *IEEE TCAD* 14(5):639–649 (1995). **[B]** — [doi:10.1109/43.384428](https://doi.org/10.1109/43.384428)
  - Odabasioglu, Celik & Pileggi, "PRIMA: passive reduced-order interconnect macromodeling algorithm," *IEEE TCAD* 17(8):645–654 (1998). It uses Arnoldi-based congruence projection and preserves passivity. **[B]** — [doi:10.1109/43.712097](https://doi.org/10.1109/43.712097)
  - Daniel, Siong, Chay, Lee & White, "A multiparameter moment-matching model-reduction approach for generating geometrically parameterized interconnect performance models," *IEEE TCAD* 23(5):678–693 (2004). This is parametric Krylov: moments are matched jointly in frequency and geometry parameters. **[B]**
- The SIAM Review survey by Benner, Gugercin & Willcox (2015) classes these methods as interpolatory projection. It points out that multipoint rational Krylov with matched points at several parameter values is equivalent to a (non-certified) RB with snapshot-plus-derivative bases. **[B]** — [doi:10.1137/130932715](https://doi.org/10.1137/130932715)
- Hochman, Fernández Villena, Polimeridis, Silveira, White & Daniel, "Reduced-order models for electromagnetic scattering problems," *IEEE Trans. Antennas Propag.* 62(6):3150–3162 (2014). **[V]**
  - Inputs are current distributions near a scatterer; outputs are the scattered fields.
  - Minimal-order models are derived via the SVD of the input-to-output operator, and DEIM makes them computable.
  - The models are reusable "scatterer macromodels" in a larger simulation.
  - [doi:10.1109/TAP.2014.2314734](https://doi.org/10.1109/TAP.2014.2314734)
- Rational approximation for sweeps:
  - Gustavsen & Semlyen, "Rational approximation of frequency domain responses by vector fitting," *IEEE Trans. Power Deliv.* 14(3):1052–1061 (1999). **[B]**
  - Nakatsukasa, Sète & Trefethen, "The AAA algorithm for rational approximation," *SIAM J. Sci. Comput.* 40(3):A1494–A1522 (2018). **[B]** — [doi:10.1137/16M1060878](https://doi.org/10.1137/16M1060878)
- Pole-expansion alternative: resonant-state/QNM methods give the S-matrix's frequency dependence analytically, so frequency sweeps are unnecessary (Weiss & Muljarov 2018, see Q4). **[V]**

### Inferences
- In partially coherent imaging the Hopkins TCC / SOCS decomposition is already an optimal (POD-like) low-rank reduction of the *illumination–pupil* operator. Krylov/RB reduction can be applied one level earlier, to the rigorous mask near-field solve across source points (oblique incidence). Known M3D (mask-3D) "field-stitching" and "domain decomposition" lithography models are informal versions of this.
- Moment matching around a nominal incidence angle is a cheap way to get the angular dependence of the mask spectrum (used in Hopkins-with-M3D corrections). It is conceptually the same as a Taylor/Padé expansion in sin θ. The search did not return a specific lithography paper on this (gap).

### Gaps
- No primary source found that applies PRIMA- or Krylov-type MOR specifically to lithography mask spectra versus illumination angle. Domain-specific lithography literature (e.g., M3D compact models) is covered by other researchers.
- Speedups for Krylov MOR in EM sweeps are problem-dependent. No single representative number was verified in this session.

---

## Q3. Empirical interpolation (EIM / DEIM / Q-DEIM) for non-affine and nonlinear parameter dependence

### Takeaway
EIM/DEIM/Q-DEIM restore the offline/online split when the operator or nonlinearity is non-affine, by interpolating the nonlinear term at a few greedily or pivot-selected "magic points." In Maxwell RB it is used for Green's kernels (BEM) and for geometric maps. For lithography it is the natural tool for the resist sigmoid, dose/threshold nonlinearities, and non-affine mask-geometry dependence. The |E|² intensity is quadratic and can be handled exactly without it.

### Cited Findings
- Barrault, Maday, Nguyen & Patera, "An 'empirical interpolation' method: application to efficient reduced-basis discretization of partial differential equations," *C. R. Math. Acad. Sci. Paris* 339(9):667–672 (2004). This is the original EIM. **[B]** — [doi:10.1016/j.crma.2004.08.006](https://doi.org/10.1016/j.crma.2004.08.006)
- Chaturantabut & Sorensen, "Nonlinear model reduction via discrete empirical interpolation," *SIAM J. Sci. Comput.* 32(5):2737–2764 (2010). DEIM uses a POD basis of nonlinear-term snapshots plus greedy index selection, so the online cost is independent of the full dimension. **[B]** — [doi:10.1137/090766498](https://doi.org/10.1137/090766498)
- Drmač & Gugercin, "A new selection operator for the discrete empirical interpolation method — improved a priori error bound and extensions," *SIAM J. Sci. Comput.* 38(2):A631–A648 (2016). Q-DEIM selects points by QR with column pivoting; it has a better error bound and is trivially parallel. **[B]** — [doi:10.1137/15M1019271](https://doi.org/10.1137/15M1019271)
- EM applications:
  - Fares et al. (2011) use EIM to make the parametrized EFIE kernel affine. **[V]** — [doi:10.1016/j.jcp.2011.03.023](https://doi.org/10.1016/j.jcp.2011.03.023)
  - Hochman et al. (2014) use DEIM to make SVD-optimal scattering ROMs computable. **[V]** — [doi:10.1109/TAP.2014.2314734](https://doi.org/10.1109/TAP.2014.2314734)
  - Hesthaven, Stamm & Zhang (2014) build efficient greedy EIM for high-dimensional parameter spaces. **[B]**

### Inferences
- The lithography forward chain is mask m → near field E(m) (linear in m under the thin mask; affine-plus-correction with M3D) → aerial image I = Σ_k |h_k ⊗ E|² (quadratic) → resist R = sigmoid(α(I − t)) (nonlinear, non-affine). Only the last step truly needs DEIM.
  - Q-DEIM sampling on the image grid would choose a few hundred "magic pixels" near edges (where the sigmoid is active). There the aerial image is computed exactly, and the resist profile is reconstructed elsewhere from a POD basis.
  - This matches the empirical observation that ILT cost and loss sensitivity concentrate at contour pixels (EPE). This is an inference; no lithography paper applying DEIM to resist models was found.
- DEIM's weakness is a missing rigorous error bound for the nonlinear term. Sharp sigmoids (large α) give poorly compressible snapshots, a Kolmogorov n-width problem similar to transport/shock problems. Localized or adaptive DEIM (local bases per contour patch) would likely be needed.

### Gaps
- No published use of EIM/DEIM/Q-DEIM in a computational lithography or resist-model context was found in this session.

---

## Q4. Quasinormal-mode (QNM) / resonant-state expansions as physics-based reduction

### Takeaway
QNM/resonant-state expansions reduce a resonator's response to a sum over a few complex-frequency eigenmodes with analytic frequency dependence. They are the most physically interpretable "ROM" for resonant nanophotonics, now with rigorous normalization, S-matrix expansions, and Riesz-projection variants. They work best for high-Q, few-mode resonators. Lithography masks (binary/PSM/EUV absorbers) are low-Q, broadband, non-resonant scatterers, apart from EUV multilayer Bragg resonance. Direct transfer is therefore limited, but the Riesz-projection machinery for *quadratic* observables and sensitivities is relevant.

### Cited Findings
- **Review:** Lalanne, Yan, Vynck, Sauvan & Hugonin, "Light interaction with photonic and plasmonic resonances," *Laser Photonics Rev.* 12, 1700113 (2018). It covers QNM normalization, completeness, and the MAN-type solvers. **[V]** — [doi:10.1002/lpor.201700113](https://doi.org/10.1002/lpor.201700113); [arXiv:1705.02433](https://arxiv.org/abs/1705.02433)
- Sauvan, Hugonin, Maksymov & Lalanne, "Theory of the spontaneous optical emission of nanosize photonic and plasmon resonators," *Phys. Rev. Lett.* 110, 237401 (2013). It introduces the PML-based QNM normalization. **[B]** — [doi:10.1103/PhysRevLett.110.237401](https://doi.org/10.1103/PhysRevLett.110.237401)
- Yan, Faggiani & Lalanne, "Rigorous modal analysis of plasmonic nanoresonators," *Phys. Rev. B* 97, 205422 (2018). It handles dispersive materials by auxiliary-field linearization, with PML modes completing the basis. **[B]**
- Recent design-oriented work: "Designing electromagnetic resonators with quasinormal modes" (2024). — [arXiv:2408.13513](https://arxiv.org/abs/2408.13513) **[V title only]**
- **Resonant-state expansion (RSE):**
  - Muljarov, Zimmermann & Langbein, "Brillouin-Wigner perturbation theory in open electromagnetic systems," *EPL* 92, 50010 (2010). It introduces a modified normalization that gives a spectral Green's function, and solves the perturbed modes as a *linear matrix eigenproblem* in the unperturbed basis. **[V]** — [doi:10.1209/0295-5075/92/50010](https://doi.org/10.1209/0295-5075/92/50010)
  - Doost, Langbein & Muljarov, *Phys. Rev. A* 85, 023835 (2012), apply RSE to planar layered systems (slabs, Bragg microcavities) and find power-law convergence in basis size. **[V via search summary]** — [arXiv:1205.4924](https://arxiv.org/abs/1205.4924)
  - Weiss & Muljarov, "How to calculate the pole expansion of the optical scattering matrix from the resonant states," *Phys. Rev. B* 98, 085433 (2018). The S-matrix pole expansion uses only resonant states, with no fitting. The S-matrix becomes analytic in wavenumber, so no frequency sweeps are needed. Demonstrated on chiral nanoantenna arrays (absorption, circular dichroism). **[V]** — [doi:10.1103/PhysRevB.98.085433](https://doi.org/10.1103/PhysRevB.98.085433)
- Alpeggiani, Parappurath, Verhagen & Kuipers, "Quasinormal-mode expansion of the scattering matrix," *Phys. Rev. X* 7, 021035 (2017). The expansion needs only complex eigenfrequencies and the far-field properties of the modes. It is accurate with overlapping modes and needs no ad-hoc nonresonant background. **[V]** — [doi:10.1103/PhysRevX.7.021035](https://doi.org/10.1103/PhysRevX.7.021035)
- **Riesz-projection variants (ZIB/JCMwave):**
  - Binkowski, Zschiedrich & Burger, "A Riesz-projection-based method for nonlinear eigenvalue problems" (2019/2020). It uses contour integration in the complex frequency plane. **[V title]** — [arXiv:1811.11624](https://arxiv.org/abs/1811.11624)
  - Binkowski et al., "Quasinormal mode expansion of optical far-field quantities," *Phys. Rev. B* 102, 035432 (2020) (volume **[B]**). It handles QNM divergence in the far field via contour integrals. **[V title/arXiv]** — [arXiv:2003.11305](https://arxiv.org/abs/2003.11305)
  - Betz, Binkowski, Hammerschmidt, Zschiedrich & Burger, "Resonance expansion of quadratic quantities with regularized quasinormal modes," *Phys. Status Solidi A* 220, 2200892 (2023). It expands **quadratic observables** (energy flux, absorption) via Riesz projection. **[V]** — [doi:10.1002/pssa.202200892](https://doi.org/10.1002/pssa.202200892); [arXiv:2212.11117](https://arxiv.org/abs/2212.11117)
  - RPExpand (*SoftwareX* 2021) is a MATLAB implementation with a FEM interface. **[V]** — [DOAJ record](https://doaj.org/article/19462888e01c44c19c4d2387d5a401cd)
  - Schneider et al., *Communications Physics* (2022). Riesz-projection eigenfrequency sensitivities used for resonator optimization. **[V via search summary]** — [EURAMET record](https://www.euramet.org/repository/research-publications-repository-link/publication/schneiderhbbzb2022)

### Inferences
- Lithography relevance:
  - **High:** EUV multilayer-mirror and absorber stacks, and resonant phase-shift structures. The response there is dominated by a few Bragg or leaky resonances, so a QNM/pole S-matrix expansion would give analytic wavelength and angle dependence. This is useful for EUV bandwidth and chief-ray-angle (CRA ≈ 6°) sweeps.
  - **Low:** binary or attenuated-PSM DUV masks, whose response is non-resonant. In that case background (PML/branch-cut) modes dominate and QNM truncation converges poorly.
- Riesz-projection expansions of *quadratic* quantities map formally onto aerial-image intensity (a quadratic form of the field). That is the closest conceptual link from QNM theory to imaging.
- The RSE "perturb within an unperturbed modal basis → small linear eigenproblem" structure resembles a reduced-basis update for mask-geometry perturbations (OPC-like edge moves).

### Gaps
- I found no paper that applies QNM/RSE expansions to photomask or lithography imaging.
- Convergence of QNM expansions for periodic gratings (Rayleigh anomalies and branch cuts) was not researched in depth here.

---

## Q5. Hierarchical matrices, ACA, FMM, low-rank compression of integral operators; tensor-train / low-rank tensor methods

### Takeaway
Fast integral-operator algebra (MLFMA; H/H²-matrices with ACA; butterfly; FFT-based VIE with Tucker/TT compression of the Green's tensor) reduces dense O(N²) EM operators to roughly O(N log N) storage and matvecs. It is complementary to parametric MOR: it compresses the *operator*, not the *parameter-to-solution map*. Tensor (Tucker/TT) compression of FFT-VIE Green's tensors is the most recent and most lithography-relevant thread, because lithography also lives on regular voxel/pixel grids with translation-invariant kernels.

### Cited Findings
- **Foundations:**
  - Hackbusch, "A sparse matrix arithmetic based on H-matrices. Part I," *Computing* 62:89–108 (1999). **[B]**
  - Bebendorf, "Approximation of boundary element matrices," *Numer. Math.* 86:565–589 (2000), which introduces ACA. **[B]**
  - Song, Lu & Chew, "Multilevel fast multipole algorithm for electromagnetic scattering by large complex objects," *IEEE TAP* 45(10):1488–1493 (1997). **[B]**
- Zhao, Vouvakis & Lee, "The adaptive cross approximation algorithm for accelerated method of moments computations of EMC problems," *IEEE Trans. EMC* 47(4):763–773 (2005). ACA is kernel-independent and algebraic, and needs only matrix-entry evaluation. **[B]**
- **Tensor compression of FFT-VIE:**
  - Giannakopoulos et al., "Memory footprint reduction for the FFT-based volume integral equation method via tensor decompositions" (arXiv 2018, IEEE TAP 2019 **[B]**). The FFT-VIE Green's-function tensors have low multilinear rank. Tucker compression lets them fit on a GPU and speeds up iterative solves. **[V arXiv]** — [arXiv:1811.00484](https://arxiv.org/abs/1811.00484)
  - Yucel group (NTU) conference papers:
    - Tucker compression of FMM-FFT translation operators gives more than 90% memory reduction with negligible overhead.
    - In the VoxCap electrostatic simulator it gives 3–4 orders of magnitude reduction of circulant-tensor memory.
    - TT-based tensor completion is used to build system tensors cheaply.
    - Tucker gives at least one order of magnitude better compression than SVD for magneto-quasi-static voxel VIE.
    - Sources: [NTU C78](https://personal.ntu.edu.sg/acyucel/conf_papers_abs/C78.pdf); [NTU C46](https://personal.ntu.edu.sg/acyucel/conf_papers_abs/C46.pdf); [Purdue SublimeLab tensor research](https://engineering.purdue.edu/SublimeLab/Research/Tensor) **[V, conference-level]**
  - A tensor-train compression scheme stores Toeplitz VIE matrices in O(log N) memory (Laplace kernel) and O(N log N) (Helmholtz kernel). — [Purdue SublimeLab](https://engineering.purdue.edu/SublimeLab/Research/Tensor) **[V summary]**
  - Corona, Rahimian & Zorin, "A tensor-train accelerated solver for integral equations in complex geometries," *J. Comput. Phys.* 334:145–169 (2017). It is a QTT-compressed hierarchical direct solver. **[B]**
- Search summaries note that Tucker suits low-dimensional Green's-function arrays, while TT/CP suit high-dimensional tensorized system matrices. No head-to-head TT-versus-Tucker benchmark exists; the evidence is mostly conference-level from one group. — [ar5iv 1811.00484](https://ar5iv.labs.arxiv.org/html/1811.00484) **[V summary]**

### Inferences
- Lithography imaging is already an FFT method: kernels are shift-invariant over the pupil and image, and SOCS gives a separable low-rank sum. Tensor (TT/Tucker) compression is most transferable in two places:
  - (i) Compressing the 4D TCC (f′, g′; f″, g″) beyond SOCS truncation, especially with polarization, defocus, and dose as extra tensor modes for process-window ILT.
  - (ii) Compressing rigorous M3D near-field libraries indexed by (source angle × polarization × pattern-context).
- H-matrices and ACA matter if rigorous mask IE/VIE solvers (for EUV absorbers) are used inside ILT. For pixel-ILT with a thin-mask or library model they are less relevant than FFT plus tensor methods.

### Gaps
- No lithography-specific paper on TT/Tucker compression of the TCC was found in this session (possibly covered by the lithography-focused researcher).
- Speedup numbers for H-matrix/ACA EM solvers in 2010–2026 (e.g., Chai & Jiao direct solvers, butterfly solvers by Michielssen and colleagues) were not verified.

---

## Q6. MOR inside optimization and inverse design (adjoint ROMs; trust-region ROM; local periodic approximation; overlapping domains)

### Takeaway
Two families matter.
- (a) **Trust-region ROM optimization** (TR-POD, Zahr & Farhat 2015, Yue & Meerbergen 2013, certified TR-RB by Qian et al. 2017) builds and refines the ROM *along the optimization path*, not globally. It gives provable convergence to the full-order optimum while cutting full-order solves by roughly 4–5× or more.
- (b) **Domain-decomposition-style physical approximations** for large-area metasurfaces: the local periodic approximation (LPA) with precomputed unit-cell response tables (Pestourie et al. 2018), and overlapping domains (Lin & Johnson 2019). These scaled adjoint inverse design to devices thousands of wavelengths wide and centimeter-scale experimental meta-optics (Li et al. 2022).

Family (b) is structurally closest to full-chip ILT, which also tiles a huge 2D domain with local, mostly-periodic patterns.

### Cited Findings
**Trust-region ROM and adjoint-based ROM optimization**
- Foundations: Arian, Fahl & Sachs, "Trust-region proper orthogonal decomposition for flow control" (ICASE Report 2000-25, 2000). This is the TR-POD framework. **[B]**
- Zahr & Farhat, "Progressive construction of a parametric reduced-order model for PDE-constrained optimization," *Int. J. Numer. Methods Eng.* 102(5):1111–1135 (2015). **[V]**
  - There is no offline/online split. Instead a sequence of reduced optimization problems is built, each trained on high-dimensional-model (HDM) samples at previous iterates.
  - A nonlinear trust region based on a *residual error indicator* keeps iterates where the ROM is accurate.
  - Sensitivities are included in the POD basis.
  - Result: **4–5× fewer HDM queries**, and the error in the optimal solution is far below 0.1% (aerodynamic shape optimization).
  - [arXiv:1407.7618](https://arxiv.org/abs/1407.7618); [bib](https://math.lbl.gov/~mjzahr/content/bib/zahr2015progressive.html)
- Yue & Meerbergen, "Accelerating optimization of parametric linear systems by model order reduction," *SIAM J. Optim.* 23(2):1344–1370 (2013). It embeds Krylov-based parametric MOR in a trust-region optimization framework for parametric linear systems, from Yue's KU Leuven PhD on MOR in design optimization. **[V citation; content B]** — [doi:10.1137/120869171](https://doi.org/10.1137/120869171)
- Qian, Grepl, Veroy & Willcox, "A certified trust region reduced basis approach to PDE-constrained optimization," *SIAM J. Sci. Comput.* 39(5):S434–S460 (2017). It uses RB a-posteriori bounds as the trust-region criterion and proves convergence. **[B]** — [doi:10.1137/16M1081981](https://doi.org/10.1137/16M1081981)
- Keil, Mechelli, Ohlberger, Schindler & Volkwein, "A non-conforming dual approach for adaptive trust-region reduced basis approximation of PDE-constrained parameter optimization," *ESAIM: M2AN* 55:1239–1269 (2021). It uses an adjoint (dual) RB space for accurate reduced gradients. **[B]**
- The Pomplun et al. (2010) RB-SMO paper is the lithography instance: an RB surrogate of the rigorous mask inside SMO, with error estimators guiding basis enrichment. **[V]** — [doi:10.1117/12.866101](https://doi.org/10.1117/12.866101)

**Large-area metasurface inverse design (Johnson / Capasso)**
- Pestourie, Pérez-Arancibia, Lin, Shin, Capasso & Johnson, "Inverse design of large-area metasurfaces," *Opt. Express* 26(26):33732–33747 (2018). **[V]**
  - The scattering problem is approximated by a composition of periodic scattering problems, one per unit cell (local periodic approximation, LPA). This gives precomputed or interpolated response tables, which are combined with near-to-far-field integrals and adjoint gradients.
  - The approach is validated against brute-force Maxwell solutions.
  - It enables surfaces thousands of wavelengths wide, multi-wavelength and multi-angle objectives, and demultiplexers.
  - It extends to non-subwavelength cells by including extra diffraction orders.
  - [doi:10.1364/OE.26.033732](https://doi.org/10.1364/OE.26.033732); [arXiv:1808.04215](https://arxiv.org/abs/1808.04215)
- Lin, Liu & Johnson, "Topology optimization of freeform large-area metasurfaces," *Opt. Express* 27(11):15765 (2019). It uses LPA-based topology optimization with 10^5–10^6 degrees of freedom. **[V]** — [doi:10.1364/OE.27.015765](https://doi.org/10.1364/OE.27.015765); [arXiv:1902.03179](https://arxiv.org/abs/1902.03179)
- Lin & Johnson, "Overlapping domains for topology optimization of large-area metasurfaces," *Opt. Express* 27(22):32445 (2019) (volume/page **[B]**). **[V]**
  - Each simulated block contains a unit cell plus overlapping neighbor regions plus PML, solved independently and in parallel.
  - It is about **10× more accurate than LPA at similar cost**.
  - Demonstrated on 200λ-diameter, NA 0.71 multichromatic and broadband-achromatic lenses with about 50% efficiency.
  - [arXiv:1906.05376](https://arxiv.org/abs/1906.05376); [DSpace@MIT](https://dspace.mit.edu/handle/1721.1/136486)
- Independent evaluation: Robben & Penninck (SPIE 2024) compare overlapping-domain analysis against LPA and full-wave FDTD for large metasurfaces. — [doi:10.1117/12.3001554](https://doi.org/10.1117/12.3001554) **[V title-level]**
- Li, Pestourie, Park, Huang, Johnson & Capasso, "Inverse design enables large-scale high-performance meta-optics reshaping virtual reality," *Nat. Commun.* 13, 2409 (2022). **[V]**
  - A fast approximate solver plus adjoint optimization makes 3D meta-optics design tractable.
  - Fabrication constraints enter through a surrogate model.
  - Experiments show centimeter-scale, high-NA, aberration-corrected visible metalenses and a VR meta-eyepiece.
  - [doi:10.1038/s41467-022-29973-3](https://doi.org/10.1038/s41467-022-29973-3); [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9064995/)
- Skarda et al., "Low-overhead distribution strategy for simulation and optimization of large-area metasurfaces," *npj Comput. Mater.* 8, 78 (2022). It distributes overlapping subdomain FDFD solves with adjoints across GPUs and CPUs. **[B]**

**ML delimitation (out of scope)**
- Later Johnson/Capasso work replaces the LPA response tables with neural or active-learning surrogates (e.g., Pestourie et al., "Active learning of deep surrogates for PDEs," *npj Comput. Mater.* 2020 **[B]**). Physics-based LPA and overlapping-domain decomposition remain the backbone; ML only replaces the table interpolation.

### Inferences
- **Direct structural analogy to ILT.**
  - The LPA is exactly the thin-mask/Kirchhoff idea generalized: each tile's transmission comes from a local periodic rigorous solve (a library), then is stitched and propagated.
  - Overlapping domains is the rigorous-tile analogue of "M3D domain decomposition with halo" models in OPC.
  - The Johnson-group pipeline of library or overlapping-domain forward model, adjoint gradient, topology optimization with filtering/projection, and imaging objective is essentially ILT with a rigorous mask model. The main differences are that the objective is a focal-spot or PSF metric rather than a resist contour, and that illumination is coherent rather than partially coherent.
  - Transfer path: replace the thin-mask model in ILT with LPA/overlapping-domain M3D tables. Keep the adjoint Hopkins/SOCS imaging chain, and apply the same filter/projection and β-continuation regularizers.
- **Trust-region ROM for ILT.** ILT already iterates on a slowly changing mask. A TR-RB/POD scheme would build a local reduced space from rigorous near fields at recent iterates, use the residual or RB bound to decide when to recompute rigorously, and call the expensive M3D solver only on trust-region failure. This could cut rigorous solves by a factor similar to Zahr & Farhat's 4–5×. That number is from aerodynamics, so lithography transfer is unproven.
- **Limits.**
  - LPA fails where neighboring cells vary quickly; the error grows with gradient in geometry, which is exactly the regime of sharp OPC serifs.
  - Overlapping domains costs about (1 + overlap/cell)^d more per cell.
  - Certified TR-RB needs affine decompositions and inf-sup bounds that are hard to obtain for 3D Maxwell with pixel-level design.

### Gaps
- I found no paper that combines LPA/overlapping-domain metasurface methods with partially coherent imaging or resist models.
- Exact speedup numbers for Pestourie 2018 versus full-wave, and parallel-scaling numbers for Skarda 2022, were not verified in this session.
- I did not confirm Yue & Meerbergen's specific test problems or speedups (paywall).

---

## Q7. Which ideas are most advanced and most transferable to lithography? (synthesis table)

### Takeaway
The most transferable physics-based ideas, ranked:
1. Trust-region / progressive ROMs inside the ILT loop (low risk, generic).
2. Library-based local periodic approximation plus overlapping-domain decomposition with adjoints (already proven at 10^5–10^6 design variables for metasurfaces).
3. Certified RBM for rigorous mask-3D over illumination and CD parameters (ZIB already did it for SMO).
4. Q-DEIM for the resist sigmoid and contour-local nonlinearities.
5. Tucker/TT compression of TCC and M3D library tensors.
6. QNM/pole expansions, only for resonant EUV multilayer/absorber stacks.

### Cited Findings (summary table; every row is sourced in Q1–Q6)

| Idea | Key refs (year, venue) | Core mechanism | Reported speed/accuracy | Main limits | Litho transfer |
|---|---|---|---|---|---|
| Certified RBM, Maxwell | Chen et al. 2010 SISC [doi](https://doi.org/10.1137/09075250X); Hess & Benner 2013 TMTT [doi](https://doi.org/10.1109/TMTT.2013.2258167) | Greedy snapshots, Galerkin, residual bounds | 50–100× order reduction; ≥200× time (Hess & Benner) | inf-sup near resonance; affine decomposition for geometry | M3D over source angle/polarization/CD |
| RBM for SMO/OPC | Pomplun et al. 2010 SPIE 7823 [doi](https://doi.org/10.1117/12.866101) | FEM offline, RB online, error estimator | "dramatically reduces" cost (no number verified) | few geometric parameters only | Direct precedent |
| RB for EFIE (BEM) | Fares et al. 2011 JCP [doi](https://doi.org/10.1016/j.jcp.2011.03.023) | RB + EIM for kernel | cheap online RCS | dense BEM offline | IE-based EUV absorber models |
| SVD+DEIM scattering macromodels | Hochman et al. 2014 TAP [doi](https://doi.org/10.1109/TAP.2014.2314734) | Minimal-order input→output ROM | not verified | per-scatterer | reusable mask-feature macromodels |
| Krylov/PRIMA, multiparameter MM | Odabasioglu 1998; Daniel 2004; survey Benner 2015 [doi](https://doi.org/10.1137/130932715) | Moment matching | — | not certified; many parameters → large basis | angle/defocus sweeps |
| EIM/DEIM/Q-DEIM | Barrault 2004; Chaturantabut & Sorensen 2010 [doi](https://doi.org/10.1137/090766498); Drmač & Gugercin 2016 [doi](https://doi.org/10.1137/15M1019271) | Interpolate nonlinear term at selected points | online cost independent of N | no rigorous nonlinear bound; sharp sigmoid → large n-width | resist sigmoid, threshold, contour-local |
| QNM/RSE/pole S-matrix | Lalanne 2018 LPR [doi](https://doi.org/10.1002/lpor.201700113); Muljarov 2010 EPL [doi](https://doi.org/10.1209/0295-5075/92/50010); Weiss & Muljarov 2018 PRB [doi](https://doi.org/10.1103/PhysRevB.98.085433); Alpeggiani 2017 PRX [doi](https://doi.org/10.1103/PhysRevX.7.021035) | Few resonant modes, analytic ω dependence | no sweep needed | low-Q/non-resonant → poor truncation | EUV multilayer and absorber resonance; bandwidth |
| Riesz projection (quadratic outputs) | Betz/Burger et al. 2023 pssa [doi](https://doi.org/10.1002/pssa.202200892) | Contour integrals for quadratic observables | — | contour-quadrature cost | intensity = quadratic form |
| FFT-VIE + Tucker/TT | Giannakopoulos et al. 2018/19 [arXiv](https://arxiv.org/abs/1811.00484); Yucel group [NTU](https://personal.ntu.edu.sg/acyucel/conf_papers_abs/C78.pdf) | Low multilinear-rank Green's tensors | >90% memory cut; Tucker ≥10× better than SVD | mostly conference evidence | TCC/M3D-library compression |
| TR-ROM optimization | Zahr & Farhat 2015 IJNME [arXiv](https://arxiv.org/abs/1407.7618); Yue & Meerbergen 2013 SIOPT [doi](https://doi.org/10.1137/120869171); Qian et al. 2017 SISC [doi](https://doi.org/10.1137/16M1081981) | ROM built along the optimization path, residual trust region | 4–5× fewer HDM solves, <0.1% optimum error (Zahr & Farhat) | needs cheap error indicator | ILT outer loop with rigorous M3D |
| LPA response tables + adjoint | Pestourie et al. 2018 OE [doi](https://doi.org/10.1364/OE.26.033732) | Unit-cell periodic solves stitched | thousands of λ wide | fails for rapidly varying neighbors | thin-mask → library-M3D in ILT |
| Overlapping domains | Lin & Johnson 2019 OE [arXiv](https://arxiv.org/abs/1906.05376) | Cell + halo + PML, parallel | ~10× more accurate than LPA at similar cost; 200λ NA 0.71 lens | overlap overhead | halo-based M3D domain decomposition |
| Scaled inverse design (experiment) | Li et al. 2022 Nat. Commun. [doi](https://doi.org/10.1038/s41467-022-29973-3) | Approximate solver + adjoint + fabrication surrogate | cm-scale metalens | surrogate fabrication model | full-field ILT analogue |

### Inferences
- The structural difference between nanophotonic inverse design and ILT is **partial coherence plus resist nonlinearity**. Nanophotonics papers almost always use coherent single or multi-wavelength objectives. ILT therefore needs (i) a reduced model per source point, or per SOCS kernel, and (ii) DEIM-type treatment of the resist. Both have mature generic tools (RB with quadratic outputs; Q-DEIM) but no lithography demonstrations were found beyond the 2009–2010 ZIB RBM work.
- Physics-based MOR offers certified error control and interpretability, which ML surrogates lack. For production lithography (sign-off accuracy, sub-nm EPE), that makes RB with a-posteriori bounds and TR-ROM attractive even where ML surrogates are faster.

### Gaps
- No quantitative head-to-head comparison (physics-based MOR versus Hopkins/SOCS thin-mask versus rigorous) for ILT was found.
- Post-2020 lithography-specific MOR publications (e.g., in *J. Micro/Nanopatterning, Materials, and Metrology* or *Proc. SPIE* Advanced Lithography) were not searched here and are expected to be covered by the lithography-focused researcher.
