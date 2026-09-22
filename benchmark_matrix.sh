#!/usr/bin/env bash
set -u

show_help() {
    cat <<'HELP'
Usage: ./benchmark_matrix.sh --vendor nvidia|intel|amd --device-name NAME --output-dir DIR [options]

Runs CPU plus each supported GPU provider in separate virtual environments.
Unavailable or failed cases are retained as JSON; summary.json compares each
successful GPU provider with CPU under the same model, batch, and input.

Options:
    --vendor NAME               GPU vendor: nvidia, intel, or amd (required)
    --device-name NAME          OS-confirmed GPU name including vendor (required)
    --output-dir DIR            Fresh result directory (required)
    --gpu-index N               CUDA/TensorRT/MIGraphX device index (default 0)
    --directml-device-index N   DirectML adapter index (Windows, default 0)
    --webgpu-device-index N     WebGPU adapter index (default 0)
    --openvino-device NAME      Intel OpenVINO device (default GPU.0)
    --profiles NAMES...         Model profiles (default wd14_v3 balanced)
    --batches NUMBERS...        Batch sizes (default 1 4)
    --providers NAMES...        Provider subset; CPU always runs
    --warmup N                 Warmup passes (default 3)
    --iterations N             Iterations per set (default 20)
    --sets N                   Number of sets (default 3)
    -h, --help                 Show this help

Example:
    ./benchmark_matrix.sh --vendor amd --device-name 'AMD Radeon Graphics' \
        --output-dir benchmarks/amd_linux_20260923
HELP
}

if [ "$#" -eq 0 ]; then
    show_help
    exit 0
fi
for argument in "$@"; do
    if [ "$argument" = "-h" ] || [ "$argument" = "--help" ]; then
        show_help
        exit 0
    fi
done

if [ -x .venv_bench_cpu/bin/python ]; then
    python_bin=.venv_bench_cpu/bin/python
else
    python_bin=python3
fi
"$python_bin" benchmark_matrix.py "$@"
