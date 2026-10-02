#!/usr/bin/env bash
# Execute providers in separate processes so their runtime packages stay isolated.
set -eu
cd "$(dirname "${BASH_SOURCE[0]}")"
export HF_HOME="$PWD/.dbv4/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_TOKEN_PATH="$HF_HOME/token"
export HF_XET_CACHE="$HF_HOME/xet"
export PYTHONUNBUFFERED=1

output_dir="${1:-benchmarks/ncnn_ultra}"
mkdir -p "$output_dir"
reference="$output_dir/cpu-b1.json"
status=0

measure() {
    local python="$1" provider="$2" batch="$3" precision="$4" name="$5"
    local args=(--provider "$provider" --profile ultra --batch-size "$batch"
                --ncnn-precision "$precision" --warmup 3 --iterations 20
                --output "$output_dir/$name.json")
    if [[ "$name" != cpu-b1 && -f "$reference" ]]; then
        args+=(--reference "$reference")
    fi
    echo "Measuring $name"
    if "$python" benchmark_ncnn.py "${args[@]}" > "$output_dir/$name.log" 2>&1; then
        tail -1 "$output_dir/$name.log"
    else
        echo "Failed: $name (see $output_dir/$name.log)"
        tail -12 "$output_dir/$name.log"
        status=1
    fi
}

for batch in 1 4; do
    measure venv_ncnn/bin/python cpu "$batch" fp32 "cpu-b$batch"
done
for precision in fp32 fp16-storage fp16-packed fp16-arithmetic; do
    for batch in 1 4; do
        measure venv_ncnn/bin/python ncnn "$batch" "$precision" "$precision-b$batch"
    done
done
if [[ -x venv_webgpu/bin/python ]]; then
    for batch in 1 4; do
        measure venv_webgpu/bin/python webgpu "$batch" fp32 "webgpu-b$batch"
    done
fi
exit "$status"
