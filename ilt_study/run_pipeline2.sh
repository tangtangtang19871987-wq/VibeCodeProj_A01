#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "bash ./run_pipeline.sh" > /dev/null; do sleep 20; done
for e in e05b_infeasible_sink; do
  echo "=== $e $(date +%H:%M:%S)" >> results/pipeline.log
  python3 experiments/$e.py > results/$e.log 2>&1 || echo "FAILED $e" >> results/pipeline.log
done
echo "=== done2 $(date +%H:%M:%S)" >> results/pipeline.log
