#!/bin/bash
# sed -i on a running bash script does not affect it; queue extra runs here.
cd "$(dirname "$0")"
while pgrep -f "bash ./run_pipeline2.sh" > /dev/null; do sleep 20; done
grep -q "=== e07b_freedom_radius" results/pipeline.log && exit 0
echo "=== e07b_freedom_radius $(date +%H:%M:%S)" >> results/pipeline.log
python3 experiments/e07b_freedom_radius.py > results/e07b_freedom_radius.log 2>&1 || echo "FAILED e07b" >> results/pipeline.log
echo "=== done3 $(date +%H:%M:%S)" >> results/pipeline.log
