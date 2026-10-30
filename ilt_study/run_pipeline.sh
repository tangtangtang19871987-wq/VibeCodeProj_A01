#!/bin/bash
# Runs the remaining experiments sequentially after E4 finishes.
cd "$(dirname "$0")"
while pgrep -f e04_global_vs_local.py > /dev/null; do sleep 20; done
for e in e05_infeasible_sink e06_gradient_cosine e07_sraf_freedom e08_relaxation_ladder \
         e09_step_weight_coupling e10_sigmoid_steepness e11_contrast_term e12_mask_corners \
         e13_adaptive_reweight e14_cost_checker_mismatch; do
  echo "=== $e $(date +%H:%M:%S)" >> results/pipeline.log
  python3 experiments/$e.py > results/$e.log 2>&1 || echo "FAILED $e" >> results/pipeline.log
done
echo "=== done $(date +%H:%M:%S)" >> results/pipeline.log
