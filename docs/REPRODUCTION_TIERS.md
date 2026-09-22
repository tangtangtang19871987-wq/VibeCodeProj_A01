# What Can and Cannot Be Reproduced Here

A per-claim map from the paper to this repository, so no reader has to infer it.

| Paper artifact | Reproducible here? | Why |
|---|---|---|
| Eq. 1 SOCS forward imaging | **YES — exact** | Public 24-kernel ICCAD13 set; verified to 1.7e-16 against independent autograd |
| Eq. 2 constant-threshold resist | **YES — exact** | Public contest parameters |
| Process corners, EPE / PVB / L2 protocol | **YES — exact** | Bit-exact against the reference contest checker on all 10 cases |
| Eq. 4 AdaIN | **YES** | Implemented and unit-tested against commanded statistics |
| Fig. 3 generator topology | **YES — structure** | Topology implemented as drawn; channel widths and block counts are **not in the paper** (G-010) |
| Eq. 5 WGAN-GP + reconstruction | **YES — form** | `l_rec`, `lambda_1`, `lambda_2` are **not in the paper** (G-020) |
| Eqs. 6-7 teacher-relative advantage | **YES** | Implemented; but the paper **contradicts itself** — Sec. 4.1 says group mean (G-012). Both implemented. |
| Eq. 8 Bernoulli/BCE policy loss | **YES — exact** | Verified equal to `torch.distributions.Bernoulli.log_prob` |
| Eq. 9 imitation loss + 25x25 smoothing | **YES — exact** | As specified |
| Eq. 10 total objective and weights | **YES — exact** | `lambda_pg=500`, `lambda_imit=1` |
| Low-resolution reward loop (8x, 100 iters) | **YES — mechanism** | Independently validated: the kernels span a fixed *physical* extent, so 8x downsampling preserves the aerial image to 0.17% of peak |
| The ILT solver itself | **NO — reimplementation** | Never specified; cited to CurvyILT (ISPD'25) with no code released (G-015) |
| **Table 1 "Ours" row** | **NO** | Requires pretraining on LithoBench (unreachable, G-041) on 8x A100 (unavailable, G-040) |
| **Table 2 "Ours (PT)" / "Ours (PT+RL)" rows** | **NO** | Same |
| StdMetal / StdContact rows | **NO** | LithoBench unreachable |
| "3x speedup" / "2x throughput" | **NO** | Wall-clock on 8x A100; this host is CPU-only, so any number would be meaningless |
| Figure 4 visualisation | **Qualitative only** | Depends on a trained generator at paper scale |

## What this repository contributes instead

1. **An exactly-verified implementation of the paper's physics and scoring
   stack**, including a correction to the public reference implementation's
   gradient (F-ADJ-01).
2. **A faithful implementation of the method** (Eqs. 4-10), with both readings of
   the paper's contradictory baseline implemented and comparable.
3. **A self-contained numerical ILT baseline on ICCAD13** measured on identical
   physics, with an iteration-budget curve that lets the paper's "roughly half
   the iteration budget" claim be examined on its own terms.
4. **An honest account** of every gap, deviation, and non-reproducible claim.
