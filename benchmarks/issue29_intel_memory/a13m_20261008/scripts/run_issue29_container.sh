set -eu
cd /experiment
OUT=/experiment/results-wsl-container
mkdir -p "$OUT"
python3 -m venv /workspace/venv_intel
/workspace/venv_intel/bin/python -m pip install numpy==2.3.5 onnxruntime-openvino==1.24.1 openvino==2025.4.1
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:/workspace/venv_intel/lib/python3.12/site-packages/openvino/libs:${LD_LIBRARY_PATH:-}
/workspace/venv_intel/bin/python inspect_issue29.py > "$OUT/container-environment.json"
cat /proc/meminfo > "$OUT/meminfo-before.txt"
cat /etc/os-release > "$OUT/os-release.txt"
uname -a > "$OUT/kernel.txt"
clinfo > "$OUT/clinfo.txt" 2>&1 || true
cat /sys/fs/cgroup/memory.max > "$OUT/cgroup-memory-max.txt"
cat /sys/fs/cgroup/memory.swap.max > "$OUT/cgroup-swap-max.txt"
/workspace/venv_intel/bin/python benchmark_intel_memory.py /model-cache/models/53e96223459a3046/model.onnx --baseline-batch-size 4 --iterations 5 --timeout 300 --output "$OUT/comparison"
