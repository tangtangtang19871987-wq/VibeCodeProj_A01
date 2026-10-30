#!/bin/bash
cd "$(dirname "$0")"
while pgrep -f "bash ./run_pipeline[45].sh" > /dev/null; do sleep 20; done
echo "=== e07c_band_islands $(date +%H:%M:%S)" >> results/pipeline.log
python3 experiments/e07c_band_islands.py > results/e07c_band_islands.log 2>&1 || echo "FAILED e07c" >> results/pipeline.log
echo "=== done6 $(date +%H:%M:%S)" >> results/pipeline.log
