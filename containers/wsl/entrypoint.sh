#!/bin/bash
# Internal container entry point; callers use run_tagger_wsl.ps1.
set -euo pipefail

# WSL host libraries plus optional vendor-specific WSL runtime overrides.
export LD_LIBRARY_PATH="/gpu-runtime:/usr/lib/wsl/lib:/opt/rocm/lib:${LD_LIBRARY_PATH:-}"

# Refresh application code, preserving Linux venvs, config and model caches.
cp /opt/dbv4/*.py /opt/dbv4/requirements.txt /opt/dbv4/run_tagger.sh /workspace/
cd /workspace
exec bash /workspace/run_tagger.sh "$@"
