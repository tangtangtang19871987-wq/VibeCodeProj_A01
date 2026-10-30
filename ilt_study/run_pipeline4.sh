#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "bash ./run_pipeline[23].sh" > /dev/null; do sleep 20; done
echo "=== e14_cost_checker_mismatch (rerun) $(date +%H:%M:%S)" >> results/pipeline.log
python3 experiments/e14_cost_checker_mismatch.py > results/e14_cost_checker_mismatch.log 2>&1 || echo "FAILED e14" >> results/pipeline.log
echo "=== done4 $(date +%H:%M:%S)" >> results/pipeline.log
