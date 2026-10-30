#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "bash ./run_pipeline[34].sh" > /dev/null; do sleep 20; done
echo "=== e15_minimax_general $(date +%H:%M:%S)" >> results/pipeline.log
python3 experiments/e15_minimax_general.py > results/e15_minimax_general.log 2>&1 || echo "FAILED e15" >> results/pipeline.log
echo "=== done5 $(date +%H:%M:%S)" >> results/pipeline.log
