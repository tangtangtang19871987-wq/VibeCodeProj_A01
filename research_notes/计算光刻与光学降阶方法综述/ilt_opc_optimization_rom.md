# Reduced-order and acceleration methods at the OPTIMIZATION level of ILT / OPC / process-window analysis (2010–2026)

Scope note: physics-based and non-ML methods. ML-ILT (Neural-ILT, GAN-OPC, DevelSet-style learned level sets, ML retargeting) is mentioned only to mark the scope boundary. Research method: web search over SPIE, IEEE/ACM, Optics Express, the CUHK (Bei Yu) and UT Austin (David Pan) paper archives, HKU/BIT repositories, and USPTO patents. Direct fetches of cuhk.edu.hk, arxiv.org and date-conference.com PDFs failed with DNS errors in this session, so several items below rest on abstracts, search-result excerpts or patent text and not on full papers. Those items are marked. Items marked "[unverified this session]" come from background knowledge and need checking before they are cited.

## Q1. Reduced parameterizations of the mask (band-limited/Fourier, level set, edge/fragment + MEEF matrix, multi-level/multigrid, hierarchical)

### Takeaway
The most advanced physics-based speedups in 2023–2026 come from two sources. The first is exploiting the optical low-pass band limit: compute and optimize only the frequencies inside the pupil support, or optimize on a coarse grid first and refine. The second is domain decomposition for full-chip tiling (multigrid-Schwarz). Classic edge-based OPC is already a reduced parameterization. Its "MEEF matrix" is literally the edge-placement Jacobian, and its diagonal dominance is what makes cheap damped fixed-point / Newton-like updates converge.

### Cited Findings
**Band-limited / Fourier-domain ILT**
- Chen W., Ma X., Zhang S., "Bandwidth-aware fast inverse lithography technology using Nesterov accelerated gradient," *Optics Express* 32(24):42639–42651 (2024), BIT. The method uses the band-limited nature of the optical system to cut computation in both the forward imaging model and the gradient (adjoint) step, and adds Nesterov acceleration to reduce the iteration count. The abstract claims "hundreds of times faster than traditional gradient-based ILT" without loss of accuracy (abstract only; derivations not checked) — [BIT Pure record](https://pure.bit.edu.cn/zh/publications/bandwidth-aware-fast-inverse-lithography-technology-using-nestero/)
- A Chinese ILT review (JEIT 2024) names computational cost as the main obstacle to wider ILT adoption and notes that frequency-domain formulations allow GPU acceleration — [JEIT "Research Progress of Inverse Lithography Technology"](https://jeit.ac.cn/article/doi/10.11999/JEIT240308)
- GPU curvy-mask ILT (arXiv 2411.07311) uses the Hopkins/SOCS model, where the aerial image is a sum of |kernel ⊛ mask|² terms computed in Fourier space, followed by a constant threshold resist — [arXiv 2411.07311](https://arxiv.org/pdf/2411.07311)

**Multi-level / multigrid ILT**
- Sun S., Yang F., Yu B., Shang L., Zeng X., "Efficient ILT via Multi-level Lithography Simulation," DAC 2023 (Fudan + CUHK). It uses a multi-level resolution method for faster lithography simulation, a down-sampling strategy for mask optimization, and an improved mask binarization function. It reports at least a 33.8% reduction in L2 loss and a 15.5% reduction in PVBand versus the prior state of the art — [CUHK PDF C163](https://www.cse.cuhk.edu.hk/~byu/papers/C163-DAC2023-Multi-ILT.pdf)
- Journal extension: "Adaptive ILT via Multi-Level Lithography Simulation," IEEE TCAD (Nov. 2025). It runs low-resolution simulation to get a cheap preliminary solution, then high-resolution simulation to fine-tune it (coarse-to-fine continuation) — [CUHK PDF J149](https://www.cse.cuhk.edu.hk/~byu/papers/J149-TCAD2025-MS-ILT.pdf)
- Sun et al., "Efficient ILT via Multigrid-Schwarz Method," DAC 2024. It handles large-layout tile assembly and stitching with a multigrid + Schwarz domain-decomposition scheme — [CUHK PDF C213](https://www.cse.cuhk.edu.hk/~byu/papers/C213-DAC2024-MS-ILT.pdf)

**Level-set ILT**
- Shen Y., Wong N., Lam E.Y., level-set ILT, *Optics Express* (2009). It poses mask synthesis as a level-set PDE solved by finite differences and reports better results than mainstream pixel-gradient methods — [HKU repository](https://repository.hku.hk/handle/10722/135098)
- Shen Y., Jia N., Wong N., Lam E.Y., "Robust level-set-based inverse lithography," *Optics Express* 19(6) (2011). It extends level-set ILT to partially coherent imaging with an expectation cost over random defocus and aberration — [HKU PDF](https://repository.hku.hk/bitstream/10722/135098/1/content.pdf)
- Geng Z. et al., level-set ILT with an added PV-band term in the cost, *J. Computer Science and Technology* (2015), DOI 10.1007/s11390-015-1549-7. It uses a hybrid conjugate-gradient method to reduce runtime — [JCST](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-015-1549-7)
- Yu Z., Chen G., Ma Y., Yu B., "A GPU-enabled level-set method for mask optimization," DATE 2021, extended in IEEE TCAD 2023. Its key contributions are GPU acceleration and an effective conjugate-gradient scheme. The TCAD paper credits Lv et al. with improving efficiency and fidelity through CG descent plus time-step adjustment — [DATE 2021 PDF](https://past.date-conference.com/proceedings-archive/2021/pdf/1022.pdf); [CUHK TCAD 2023 PDF](https://www.cse.cuhk.edu.hk/~byu/papers/J80-TCAD2023-LevelSet.pdf); [DATE 2021 slides](https://www.cse.cuhk.edu.hk/~byu/papers/C115-DATE2021-LevelSet-slides.pdf)

**Edge/fragment OPC, the MEEF matrix and cross-MEEF**
- The MEEF matrix captures proximity interactions between fragments. Its off-diagonal terms ("cross-MEEF", or edge-fragment correlation) give the wafer-edge response to a neighbor's mask-edge move. It is usually diagonally dominant because self-enhancement exceeds cross-enhancement (Mentor patents) — [US 8539391 "Edge fragment correlation determination for OPC"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8539391); [US 8881070](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8881070); [US 8910098 "Neighbor-aware edge fragment adjustment"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8910098)
- Convergence control patent: damping each correction to roughly half of the local-MEEF-estimated full step improves convergence — [US 8656319 "OPC convergence control"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8656319); see also [US 10437950](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10437950)
- DiffOPC (CUHK, ICCAD 2024) defines MEEF as the Jacobian of EPE with respect to fragment displacements, with the linear model e ≈ e₀ + M·d. MEEF-based OPC uses Newton-type steps with M **pre-computed and frozen** during optimization. DiffOPC instead differentiates through the edge-based forward model — [ICCAD 2024 DiffOPC slides](https://webarchive.lib.cuhk.edu.hk/20241030001211mp_/http:/www.cse.cuhk.edu.hk/~byu/papers/C240-ICCAD2024-DiffOPC-slides.pdf)

### Inferences
- For a "many recipes on a fixed layout" workflow, a frozen MEEF matrix M is the ready-made reduced Jacobian. A change in dose or threshold shifts the EPE vector e₀ (computable from one image), and the fragment correction is Δd ≈ −M⁻¹Δe₀. Because M is diagonally dominant, a few Jacobi/damped iterations suffice. This gives a first-order "re-OPC without re-OPC" and is the same logic production tools use for damped iterations.
- Band-limiting and multi-level resolution are complementary. Band-limiting reduces the per-iteration cost, because the mask only matters within roughly 2NA/λ·(1+σ). Multi-level warm-starts the fine solve. Both carry over directly to warm-starting a recipe sweep: solve the nominal recipe at full resolution, then re-optimize perturbed recipes at a coarse level and refine only the residual.

### Gaps
- Full speed/accuracy tables (runtime in seconds and speedup factors) for Multi-ILT, MS-ILT and the TCAD 2025 version could not be read (the CUHK site was unreachable in this session). Only the abstract-level L2 and PVBand improvements are confirmed.
- I found no peer-reviewed description of "hierarchical ILT / pixel-grouping" beyond multi-level down-sampling. Commercial curvilinear ILT tools (Synopsys Proteus ILT, Siemens Calibre curvilinear ILT/pxOPC) are known to use GPU and hierarchical tiling, but I found no technical paper with numbers. [unverified this session]
- Pang L., "Inverse lithography technology: 30 years from concept to practical, full-chip reality," *JM3* 20(3) 030901 (2021), DOI 10.1117/1.JMM.20.3.030901, is a likely primary industry review. [unverified this session]

## Q2. Low-rank Jacobian / Gauss-Newton / Hessian-free / Newton methods; warm start, continuation; parametric sensitivity of the optimum

### Takeaway
Production and academic ILT is dominated by first-order methods: steepest descent, CG, Nesterov, Adam-like. Full Newton is rejected because of ill-conditioning and size. Hessian-free CG (patented) and Hutchinson-diagonal second-order methods (DATE 2026) are the frontier. I found **no** ILT/OPC paper that does implicit-function-theorem / parametric-sensitivity updates of the optimal mask with respect to dose, threshold or focus. This is a genuine gap the user could exploit.

### Cited Findings
- Full Newton is unsuitable. The non-smooth objective makes the Hessian ill-conditioned, and the variable count (about 10⁶ pixels) makes it prohibitive to store or compute. CG converges faster than steepest descent — [Chan & Lam (Purdue) mask optimization paper, 2008](https://engineering.purdue.edu/ChanGroup/publication/C2008_Chan_Lam.pdf) (pre-2010 background)
- Hessian-free CG mask optimization patent. L-BFGS approximations "may fail to capture" the objective's curvature, which slows convergence. It claims a Hessian-free conjugate-gradient method based on Hessian-vector products — [US 11366952 (Google Patents)](https://patents.google.com/patent/US11366952); [Justia](https://patents.justia.com/patent/11366952)
- "Scalable second order optimizer for full chip inverse lithography," DATE 2026 (HKBU co-authors). It argues that first-order optimizers need hundreds of iterations, uses curvature information, and approximates the Hessian diagonal with Hutchinson's stochastic estimator to avoid exact Hessians — [HKBU scholars record](https://scholars.hkbu.edu.hk/en/publications/scalable-second-order-optimizer-for-full-chip-inverse-lithography/); [DATE 2026 PDF](https://past.date-conference.com/proceedings-archive/2026/DATA/531.pdf)
- Nesterov acceleration in band-limited ILT reduces the iteration count (Chen, Ma, Zhang 2024) — [BIT Pure](https://pure.bit.edu.cn/zh/publications/bandwidth-aware-fast-inverse-lithography-technology-using-nestero/)
- Coarse-to-fine continuation (low-resolution preliminary solve followed by high-resolution fine-tuning) is the core of Adaptive Multi-level ILT, TCAD 2025 — [CUHK J149](https://www.cse.cuhk.edu.hk/~byu/papers/J149-TCAD2025-MS-ILT.pdf)
- MEEF-matrix OPC (Newton on fragments with a frozen Jacobian) is the established reduced Gauss-Newton scheme — [DiffOPC slides](https://webarchive.lib.cuhk.edu.hk/20241030001211mp_/http:/www.cse.cuhk.edu.hk/~byu/papers/C240-ICCAD2024-DiffOPC-slides.pdf)
- MOSAIC (DAC 2014) starts gradient descent from a mask seeded with rule-based SRAFs, which is a warm start from rule-based OPC — [DAC 2014 MOSAIC slides](https://webarchive.lib.cuhk.edu.hk/20221209070306mp_/http:/www.cse.cuhk.edu.hk/~byu/papers/C26-DAC2014-MOSAIC-slides.pdf)

### Inferences
- **Low-rank structure is built into the forward model.** The SOCS/Hopkins decomposition truncates the TCC to K kernels (typically about 10–30). The Gauss-Newton Jacobian of the aerial image, J = Σ_k 2Re{conj(h_k⊛m)·(h_k⊛·)}, therefore has a band-limited range of dimension at most about K·N_pupil-frequencies, far smaller than the number of mask pixels. This explains why the bandwidth-aware ILT speedups (computing only in-band frequencies) are large, and it suggests a natural reduced basis: the in-band Fourier coefficients of the mask.
- **Implicit-function-theorem recipe update (proposal, not found in the literature).** At an optimum m*(p) of F(m;p) with p = (dose, threshold, focus, bias), dm*/dp = −H⁻¹ ∂²F/∂m∂p. H⁻¹v can be computed with a few Hessian-free CG iterations using the same adjoint machinery as the HF patent above, or with the Hutchinson diagonal as a preconditioner. A dose change is a scalar multiple of intensity, and a constant-threshold change is equivalent to a dose change. So for dose and threshold, ∂²F/∂m∂p is one extra adjoint solve, which gives a first-order optimal-mask predictor per recipe at about the cost of 1–2 ILT iterations.
- The sigmoid-resist ILT loss is invariant under (dose, threshold) → (α·dose, α·threshold). Recipe sweeps over dose and threshold therefore collapse to a single effective parameter t_r = threshold/dose for the aerial-image-threshold model. This halves the sweep dimension.

### Gaps
- No ILT/OPC paper found on parametric sensitivity of the optimum, IFT-based mask prediction, or warm-starting across a recipe sweep. Searches returned nothing.
- The speed/accuracy numbers of the DATE 2026 second-order paper could not be retrieved (the PDF host was unreachable).
- No head-to-head Newton vs. Gauss-Newton vs. CG benchmark for ILT was found.

## Q3. Process-window-aware OPC/ILT: cheap handling of focus/dose variation

### Takeaway
Three cheap mechanisms dominate. (i) **Few-corner PV-band objectives**: MOSAIC and the ICCAD-2013 contest use nominal plus inner/outer corners, i.e. ±dose and ±defocus. (ii) **Expectation / stochastic-gradient objectives** over random focus and dose (Jia & Lam 2010; Shen et al. 2011; Ma/Li vector-model robust optimization) sample one corner per iteration instead of all corners. (iii) **Taylor or analytic expansion of the image in defocus/dose**. Defocus enters through a quadratic pupil phase, so the expansion has no linear term near best focus, and a few simulations interpolate the whole window (ASML/Brion patents; UT Austin 2006, which reports only about 2–3× the single-focus runtime).

### Cited Findings
- Gao J.-R., Xu X., Yu B., Pan D.Z., "MOSAIC: Mask optimizing solution with process window aware inverse correction," DAC 2014, DOI 10.1145/2593069.2593163. It is a gradient ILT that jointly optimizes nominal fidelity and PV-band across process corners, with "MOSAIC_fast" and "MOSAIC_exact" modes. It was evaluated on 32 nm ICCAD-2013 contest designs and starts from rule-based SRAFs — [Unpaywall DOI record](https://unpaywall.org/10.1145%2F2593069.2593163); [UT Austin PDF](https://www.cerc.utexas.edu/utda/publications/C163.pdf); [slides](https://webarchive.lib.cuhk.edu.hk/20221209070306mp_/http:/www.cse.cuhk.edu.hk/~byu/papers/C26-DAC2014-MOSAIC-slides.pdf)
- Jia N., Lam E.Y., "Machine learning for inverse lithography: using stochastic gradient descent for robust photomask synthesis," *J. Opt. A: Pure Appl. Opt.* 12(4):045601 (2010). It treats focus variation as a random variable and uses SGD so that the per-iteration cost stays that of one focus condition. It is "ML" only in name (SGD on a physics model). Conference version: ICIP 2010, pp. 4173–4176 — [HKU hub 10722/124679](https://hub.hku.hk/handle/10722/124679); [HKU hub 10722/158674](https://hub.hku.hk/handle/10722/158674)
- Shen et al. 2011, robust level-set ILT with an expectation cost over defocus and aberration — [HKU PDF](https://repository.hku.hk/bitstream/10722/135098/1/content.pdf)
- BIT (Ma X., Li Y. et al.), "Robust resolution enhancement optimization methods to process variations based on vector imaging model" — [BIT Pure](https://pure.bit.edu.cn/en/publications/robust-resolution-enhancement-optimization-methods-to-process-var/)
- Geng et al. 2015 add a PV-band term to level-set ILT — [JCST](https://jcst.ict.ac.cn/article/doi/10.1007/s11390-015-1549-7)
- Taylor expansion in defocus. Without strong aberrations the through-focus intensity is symmetric, so first-order terms are negligible and second-order terms dominate. The pupil defocus phase expansion "does not contain a linear term". A few simulations per layout are parameterized so that the whole process window is covered by smooth interpolation, which matters for iterative PW-aware OPC (ASML/Brion patents) — [US 8527255](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8527255); [US 8200468](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8200468); [US 9390206](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9390206); calibration variant [US 8418088](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8418088); [US 9009647](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9009647)
- PW-maximizing OPC patent — [US 9360766](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9360766)
- UT Austin (Pan group) 2006 analytic defocus aerial-image formula. It computes images at any defocus with only about a 2–3× runtime increase over single-focus simulation, validated against PROLITH — [CERC PDF](https://www.cerc.utexas.edu/utda/publications/2006_fast_litho_simulation_for_design.pdf)
- A second-order polynomial-in-defocus aerial-image approximation is used to study first- and second-order defocus effects across RETs (BIT, CIOP 2021, SPIE) — [BIT Pure](https://pure.bit.edu.cn/en/publications/simulations-of-impacts-on-aerial-image-performance-of-defocus-on-/)

### Inferences
- For the user's recipe sweep, image models that are separable in focus and dose are the key trick. Dose is an exact scalar on intensity: I(x;dose) = dose·I₀(x). Defocus is captured by I(x;f) ≈ I₀ + f²·I₂ (+ f·I₁ with aberrations), where I₀, I₁, I₂ each cost one SOCS-like simulation (focus-derivative kernels). Every (dose, focus, threshold) corner then costs O(N) pointwise algebra instead of a full simulation, and PV-bands for hundreds of corners come almost free.
- SGD over corners (Jia & Lam) is the optimization-level analogue: the cost per iteration is constant regardless of the number of corners.

### Gaps
- Few-corner vs. dense PV-band accuracy statistics were not found in the sources reached.
- The validity range of the second-order focus Taylor model (in nm of defocus for a given NA/λ) was not quantified in the sources retrieved.

## Q4. Resist model reduction (VTR, CM1, Gaussian-diffused threshold) and fast evaluation of resist recipe changes

### Takeaway
Compact resist models are already reduced-order. VTR/VT5 compute a threshold that varies with local image features (peak intensity, slope, ...). CM1 is a linear-in-coefficients "variable dose contour" response-surface model, which is easy to calibrate and has long-range loading terms. Because they act pointwise or with small convolutions on a cached aerial image, a resist recipe change (new coefficients, threshold, diffusion length) can be re-evaluated without recomputing optics. Re-OPC is needed only when the target edges move.

### Cited Findings
- VTR origin: Cobb & Zakhor (SPIE 2726, 1996) model EPE as a function of peak local image intensity and cutline slope. VTRE (1999) and VT-5 (2002) were production enhancements. The 2002 study used 14 company process setups and about 3000 CDs, and found that models with too many eigenvectors overfit CD curves — [SPIE 4691 "Universal process modeling with VTRE for OPC"](https://spiedigitallibrary.org/conference-proceedings-of-spie/4691/0000/Universal-process-modeling-with-VTRE-for-OPC/10.1117/12.474587.full); [SPIE 3051 "Experimental results on OPC with variable threshold resist"](https://medicalimaging.spiedigitallibrary.org/conference-proceedings-of-spie/3051/0000/Experimental-results-on-optical-proximity-correction-with-variable-threshold-resist/10.1117/12.275977.full)
- VTR = constant threshold + correction polynomial, calibrated by best fit to measured CDs — [US 7900169 "OPC model calibration process"](https://patents.google.com/patent/US7900169)
- CM1 is a "variable contour dose" response-surface model with an adjustable threshold, while VT5 is a variable-threshold model with an adjustable response surface — [US 10120963](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/10120963)
- Granik, Cobb, Medvedev (SPIE 6154, 2006). CM1 is easier to calibrate because it is linear in its coefficients, and it includes long-range loading terms — [SPIE 6154 61543E "Application of CM0 resist model to OPC and verification"](https://medicalimaging.spiedigitallibrary.org/conference-proceedings-of-spie/6154/61543E/Application-of-CM0-resist-model-to-OPC-and-verification/10.1117/12.656566.full)
- Limitation: compact models lack vertical resist physics (z-diffusion, out-diffusion), so a separate threshold per resist height often fits better — [Academia.edu "Process Model Accuracy Enhancement Using Cluster Based Approach"](https://www.academia.edu/170136867/Process_Model_Accuracy_Enhancement_Using_Cluster_Based_Approach)

### Inferences
- A Gaussian-diffused threshold model (I_r = G_σ ⊛ I, contour at t_r) needs only one extra FFT-multiply per σ. Because G_σ is band-limited and multiplies the TCC, it can be absorbed into the SOCS kernels (h_k → G_σ-filtered TCC), so a sweep over σ costs one kernel re-decomposition per σ, or one Taylor term in σ². For a fixed σ, a threshold or dose sweep is pure post-processing of a cached image.
- In an ILT objective, a VTR/CM1-type resist adds a cheap, pointwise, differentiable map. The adjoint gradient therefore changes only by a pointwise factor, which keeps the IFT update in Q2 cheap.

### Gaps
- No 2010–2026 paper was found that systematically benchmarks the speed of re-evaluating resist recipe changes. Most VTR/CM1 literature predates 2010 and is paywalled.

## Q5. Retargeting / bias tables / model-based retargeting: handling threshold and dose shifts without re-OPC

### Takeaway
Industry handles target-level changes with rule-table retargeting (selective bias/shift, Lucas et al. 2003), then moved to model-based retargeting (Yang et al. 2009), which iterates retarget → OPC → PW check. ILT co-optimization is reserved for hotspots because it is costly. ML retargeting trained on ILT data is the recent (out-of-scope) step. I found no published industry method that absorbs a threshold or dose shift without re-running OPC beyond global or edge biasing.

### Cited Findings
- Yang E., Li C.H., Kang X.H., Guo E., "Model-based retarget for 45nm node and beyond," Proc. SPIE 7274, 727428 (2009), DOI 10.1117/12.814085. It replaces table-driven retargeting with process-window simulation per layout, resizing failing features and iterating retarget-OPC-retarget until the windows pass — [SPIE](https://proceedings.spiedigitallibrary.org/conference-proceedings-of-spie/7274/727428/Model-based-retarget-for-45nm-node-and-beyond/10.1117/12.814085.full)
- Rule tables become ineffective as retargeting grows complex. ILT gives better PW but is limited to small areas (hotspot repair) by cost. A follow-up uses ILT co-optimization to train an ML target predictor for line/space layers (ML, scope boundary) — [SPIE author profile listing](https://astronomicaltelescopes.spiedigitallibrary.org/profile/Marco.Guajardo-4059412)
- Rule-based retargeting (Lucas et al., SPIE 5040, 2003) applies selective biases and shifts to target edges while OPC stays computed at nominal conditions — [US 12411422 "Rule-based retargeting of target pattern"](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12411422)
- DATE 2012: aerial-image-based model retargeting for 28 nm metal that shifts and resizes hotspots at tape-out — [DATE 2012 12.7.2](https://past.date-conference.com/proceedings-archive/2012/PDFFILES/12.7_2.PDF)
- Defect-probability-based process window and simulation calibration patents (ASML) — [US 11079687](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11079687); [US 12386268](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12386268)

### Inferences
- A uniform dose or threshold shift moves every printed edge by roughly ΔCD/2 ≈ −(Δt/|∇I|) at each edge. The required mask-edge correction is then Δd = M⁻¹·(that EPE shift), using the frozen MEEF matrix from Q1. This is the physics-based "bias table generated on the fly" and needs no new optical simulation beyond the cached image and its slope (the image log-slope, ILS). It is valid while |Δt| keeps the edges within the linear MEEF range (a few nm).

### Gaps
- Actual commercial bias-table formats and Synopsys/Siemens retargeting algorithm details are not public in the sources reached (paywalled or proprietary).

## Q6. POD / reduced-basis / snapshot methods for ILT/OPC and the PDE-constrained ROM-optimization literature ILT could borrow

### Takeaway
I found **no** published ILT/OPC work using POD, reduced bases, or trust-region ROM optimization. In general PDE-constrained optimization, the mature recipe is the trust-region ROM (TR-POD / TR-RB): build a cheap ROM from snapshots, optimize it inside a trust region governed by an error indicator, then enrich and repeat. Zahr & Farhat report a 4–5× reduction in high-fidelity queries. This is the main transferable idea for recipe sweeps on a fixed layout.

### Cited Findings
- Zahr M.J., Farhat C., "Progressive construction of a parametric reduced-order model for PDE-constrained optimization" (arXiv 1407.7618, 2014; journal version in *Int. J. Numer. Meth. Engng.*, 2015). It solves a sequence of reduced optimization problems with a nonlinear trust region driven by a residual-based error indicator, so the trajectory stays where the ROM is accurate. It reduces high-dimensional-model (HDM) queries by a factor of 4–5 on subsonic aerodynamic shape optimization — [arXiv 1407.7618](https://arxiv.org/pdf/1407.7618); [ar5iv HTML](https://ar5iv.arxiv.org/html/1407.7618)
- Zahr PhD thesis (Stanford 2016), "Adaptive model reduction to accelerate optimization problems governed by PDEs". Its error-aware trust-region POD finds the optimal shape of a full aircraft in about half the time of standard methods — [Stanford PURL](https://purl.stanford.edu/bh989ww6442); [thesis PDF](https://math.lbl.gov/~mjzahr/content/thesis.pdf); [bib](https://math.lbl.gov/~mjzahr/content/bib/zahr2016phd_bib.html)
- Aguiló (Sandia), trust-region algorithm with ROM for bound-constrained PDE optimization. It detects loss of ROM predictive accuracy, automatically updates the POD basis, uses projected gradients for bound constraints (relevant to box-constrained mask pixels 0 ≤ m ≤ 1) and is demonstrated on topology optimization. It credits Fahl (TR-POD framework with adaptive basis updates) and Kragel (multilevel extension) — [OSTI 1512909](https://www.osti.gov/biblio/1512909); [engrXiv preprint](https://engrxiv.org/preprint/view/7218)
- ALCF talk "Accelerating PDE-constrained optimization problems using adaptive ROMs" (Zahr) — [ALCF](https://www.alcf.anl.gov/events/accelerating-pde-constrained-optimization-problems-using-adaptive-reduced-order-models)

### Inferences
- Mapping to ILT. The "PDE" is the SOCS forward model, which is already cheap per evaluation. The expensive part is the **number of optimizer iterations × recipes**, so the ROM should target the **optimization** variable. Collect snapshots of optimal masks m*(p_i) (or their in-band Fourier coefficients) for a few recipes p_i, form a POD basis Φ (expected to be low-dimensional because masks for nearby recipes differ mainly by edge shifts and SRAF intensity), then solve min_a F(Φa; p) for new recipes in a trust region. The full-order gradient ∇F(Φa) serves as the error indicator, followed by full-order polishing if needed. This is TR-POD (Fahl / Arian–Fahl–Sachs) with the mask as the control.
- A topology-optimization analogy holds: ILT pixel masks with a sigmoid/binarization projection are structurally the same as density-based topology optimization, so the Aguiló projected-gradient TR-ROM is the closest template.
- Edge-sensitive caveat: POD over binary-like masks handles translations of edges poorly (the Kolmogorov n-width problem for transport). Working in level-set φ or in band-limited Fourier coefficients, where edge shifts are smooth, should give much better compressibility than pixel masks.

### Gaps
- Kunisch & Volkwein (Galerkin POD, Numer. Math. 2001; POD-based SQP / a-posteriori error estimates), Arian–Fahl–Sachs (trust-region POD, 2000), Qian–Grepl–Veroy–Willcox ("A certified trust region reduced basis approach to PDE-constrained optimization," SIAM J. Sci. Comput. 2017) and Keil–Mechelli–Ohlberger–Schindler–Volkwein (adaptive TR-RB, ESAIM:M2AN 2021) are the key general references. They were **not retrieved in this session** and need verification of DOIs. [unverified this session]
- No evidence was found of any EDA-vendor or academic POD/RB work for OPC/ILT, even though searches targeted it. The absence is itself a finding: it marks open territory.

---
### Cross-cutting summary of the most advanced physics-based tricks (for the report writer)
1. Band-limited (in-pupil Fourier) mask/gradient computation plus Nesterov: claimed "hundreds of times" faster (Chen/Ma/Zhang, OE 2024).
2. Coarse-to-fine multi-level ILT (DAC'23 / TCAD'25) and multigrid-Schwarz tiling (DAC'24): ≥33.8% L2 and 15.5% PVBand gains at lower runtime.
3. Second-order: Hessian-free CG (patent US 11366952) and Hutchinson-diagonal curvature (DATE 2026).
4. Frozen MEEF-matrix (edge Jacobian) damped Newton for fragment OPC. It is diagonally dominant, so damping at about 0.5 works.
5. PW handling: few-corner PV-band (MOSAIC), SGD over corners (Jia & Lam 2010), even-order Taylor in defocus plus an exact linear dose scaling (ASML patents; UT Austin 2006, 2–3× single-focus cost).
6. Compact resist (VTR/VT5/CM1) applied as post-processing on cached images, so a resist recipe change needs no new optics.
7. Open opportunities not found in the literature: IFT / parametric-sensitivity updates of the optimal mask across recipes, and TR-POD/RB over mask snapshots.
