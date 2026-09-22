# Source Inventory

Status date: 2026-09-22. Session network policy: outbound HTTPS is filtered by an
organization egress proxy. Every retrieval attempt below is logged with its exact
outcome so that the provenance of every claim in this repository is auditable.

## S0. Target paper (PRIMARY — NOT OBTAINED)

| Field | Value |
|---|---|
| Title | Pushing the Limits of Inverse Lithography with Generative Reinforcement Learning |
| Authors | Haoyu Yang, Haoxing Ren (NVIDIA) |
| Venue | 63rd Design Automation Conference (DAC'26); also presented at SPIE Advanced Lithography + Patterning 2026, paper 13980-26 |
| arXiv | 2602.19027v1 |
| Primary URL | https://arxiv.org/html/2602.19027v1 |

**The primary PDF/HTML could not be retrieved in this session.**

### Retrieval attempts (all verbatim outcomes)

| # | Target | Tool | Outcome |
|---|---|---|---|
| 1 | `arxiv.org/html/2602.19027v1` | WebFetch | `EGRESS_BLOCKED` — "Access to arxiv.org is blocked by the network egress proxy" |
| 2 | `arxiv.org/abs/2602.19027` | curl | `CONNECT tunnel failed, response 403` |
| 3 | `www.alphaxiv.org/abs/2602.19027` | curl | 403 |
| 4 | `api.semanticscholar.org` (graph API) | curl | 403 |
| 5 | `export.arxiv.org` | curl | 403 |
| 6 | `ar5iv.labs.arxiv.org` | curl | 403 |
| 7 | `huggingface.co/papers/2602.19027` | curl | 403 |
| 8 | `awesomepapers.io/.../2602.19027` | WebFetch | `EGRESS_BLOCKED` |
| 9 | `spie.org/.../13980-26` | WebFetch | `EGRESS_BLOCKED` |
| 10 | `dl.acm.org`, `openreview.net`, `paperswithcode.com`, `research.nvidia.com` | curl | 403 |

Per the environment's proxy policy ("do not retry or route around organization
policy denials"), no circumvention was attempted. Only `github.com` /
`raw.githubusercontent.com` and the managed WebSearch tool were reachable.

**Consequence:** no claim in this repository is sourced to the primary text.
Everything attributed to the paper is marked with a confidence level and a
secondary-source citation. See `docs/gap_ledger.md`.

## S1. Secondary sources actually obtained

| ID | Source | Type | Reliability | What it gave |
|---|---|---|---|---|
| S1.1 | Anthropic WebSearch summaries of the arXiv abstract page (4 independent queries) | Machine summary of primary abstract | **Medium-High** for abstract-level claims; the abstract text was paraphrased consistently across queries | Abstract content, headline results (">20% EPE improvement on ICCAD13", "2–3x speedup", "half the iteration budget"), venue, authors |
| S1.2 | `memgrafter/research-digests` @ `ml_research_analysis_2026/2602.19027_*.md` | LLM-generated paper digest (generator model recorded in its front-matter as `openrouter/stepfun/step-3.5-flash:free`) | **Low-Medium.** Unverified third-party LLM reading of the paper. Quotes section numbers (3.2, 3.2.1, 3.3.2) and specific hyperparameters, but an LLM digest can hallucinate numbers. | Architecture sketch (style-aware U-Net, AdaIN at coarse level), two-stage training, reward definition, claimed hyperparameters (K=16, 8x downsample, ~100 ILT steps, lambda_imit=1, lambda_pg=500, z-dim=256), one claimed data point (StdContact avg EPE 9.0 -> 6.5) |
| S1.3 | `OpenLithoHub/OpenLithoHub` @ `src/openlithohub/models/grpo_warm_start.py` | Third-party code citing this arXiv ID | **Very Low as a reproduction reference.** Its own docstring says "Clean-room implementation of GRPO mechanism"; it wraps a VAE, uses `group_size=4` and PPO-style ratio clipping, which contradicts S1.2's description. **Not used as an implementation reference.** | Evidence only that the paper exists and is being cited |

## S2. Public assets that ARE fully obtained and verified

These carry the reproduction. They are public, version-pinned, and checksummed.

| ID | Asset | Source | Verified in this session |
|---|---|---|---|
| S2.1 | ICCAD 2013 CAD Contest Problem-1 benchmark, 10 M1 layouts (`M1_test1..10.glp`) | `github.com/OpenOPC/OpenILT` @ `benchmark/ICCAD2013/` | Parsed all 10; polygon counts and bounding boxes confirmed |
| S2.2 | 24-term SOCS optical kernels, focus + defocus, plus conjugate-transpose (CT) and combo variants | same repo, `kernel/kernels/*.pt` | Shapes `(35,35,24)` complex64; eigenvalue scales strictly descending (86.94, 35.42, 35.41, 14.17, 11.24, ...) |
| S2.3 | Reference Hopkins/SOCS forward model with analytic adjoint | same repo, `pylitho/exact.py` | Executed; open-field intensity = **0.95154** (4.8% SOCS truncation loss), 400 nm square center intensity = **0.93562** |
| S2.4 | Reference ICCAD13 evaluation protocol (L2, PV Band, EPE) | same repo, `pyilt/evaluation.py` | Executed on all 10 cases; constants `EPE_CONSTRAINT=15`, `EPE_CHECK_INTERVEL=40`, `MIN_EPE_CHECK_LENGTH=80` |
| S2.5 | ICCAD13 process configuration | same repo, `config/lithoiccad13.txt` | `KernelNum 24`, `TargetDensity 0.225`, `PrintThresh 0.5`, `PrintSteepness 50.0`, `DoseMax/Nom/Min 1.02/1.00/0.98` |

`OpenILT` is Apache-2.0 licensed. Its role here is as the **public reference
implementation of the ICCAD13 lithography model and scoring protocol** that this
paper (and essentially all ILT papers in this line) is benchmarked against. It is
not the paper's code and does not contain the paper's method.

## S3. Assets NOT obtained

| Asset | Why it matters | Status |
|---|---|---|
| Official author code for 2602.19027 | Would settle every architecture/hyperparameter gap | Searched GitHub code+repo search; **no release found**. Consistent with the paper being recent and from an industrial lab. |
| LithoBench dataset (MetalSet / ViaSet / StdMetal / StdContact) | The paper's main training and Table-2 evaluation set (per S1.1/S1.2) | Host unreachable under this egress policy. Not downloadable here. |
| Paper's Tables 1 and 2 raw numbers | Required for the L2 quantitative comparison | Not obtainable from any reachable source; WebSearch explicitly failed to surface them across 3 targeted queries |
| Paper's supplementary material | Unknown whether one exists | Cannot determine without the primary source |
