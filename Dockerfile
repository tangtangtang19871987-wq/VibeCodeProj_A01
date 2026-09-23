# CPU-only image. No CUDA anywhere -- this project runs entirely on CPU by
# explicit design (see docs/environment_audit.md); do not add a GPU base image
# or CUDA packages without updating that documentation to match reality.
FROM python:3.11-slim

# git is required by scripts/fetch_data.sh (clones the pinned OpenILT commit
# for the public ICCAD13 kernels and benchmark layouts).
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Fetches ~1MB of public kernels/benchmark layouts and verifies checksums.
# Requires network access at build time; if that is unavailable, skip this
# line and run `./scripts/fetch_data.sh` once inside a running container
# instead (it is idempotent and safe to re-run).
RUN ./scripts/fetch_data.sh

# Runs the full test suite by default -- a clean-environment correctness
# check, not a long-running service. Override the command to run an
# experiment instead, e.g.:
#   docker run --rm gril python -m gril.experiments.run_iccad13 \
#       --config configs/experiments/iccad13_ilt.yaml
ENV PYTHONPATH=/app/src
CMD ["python", "-m", "pytest", "-q"]
