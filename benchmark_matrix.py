"""Run comparable ONNX Runtime provider cases without hiding failed backends."""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

MATRIX = {
    "Windows": {
        "nvidia": ("cuda", "tensorrt", "directml", "webgpu"),
        "intel": ("intel", "directml", "webgpu"),
        "amd": ("directml", "webgpu"),
    },
    "Linux": {
        "nvidia": ("cuda", "tensorrt", "webgpu"),
        "intel": ("intel", "webgpu"),
        "amd": ("migraphx", "webgpu"),
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor", choices=("nvidia", "intel", "amd"), required=True)
    parser.add_argument("--device-name", required=True,
                        help="GPU name confirmed by the operator using OS/device tools")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0,
                        help="CUDA/TensorRT/DirectML/MIGraphX device index")
    parser.add_argument("--directml-device-index", type=int, default=0)
    parser.add_argument("--webgpu-device-index", type=int, default=0)
    parser.add_argument("--openvino-device", default="GPU.0")
    parser.add_argument("--profiles", nargs="+",
                        default=["wd14_v3", "balanced"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--providers", nargs="+",
                        help="Subset of the platform/vendor providers; cpu is always included")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sets", type=int, default=3)
    args = parser.parse_args()
    from dbv4 import MODEL_PROFILES

    invalid_profiles = [name for name in args.profiles if name not in MODEL_PROFILES]
    if invalid_profiles:
        parser.error(f"unknown profile(s): {invalid_profiles}")
    system = platform.system()
    if system not in MATRIX:
        parser.error(f"unsupported OS: {system}")
    if args.vendor not in args.device_name.lower():
        parser.error("--device-name must include --vendor; confirm the actual adapter")
    if min(*args.batches, args.warmup, args.iterations, args.sets) < 1:
        parser.error("batch and repetition counts must be positive")
    providers = ("cpu",) + MATRIX[system][args.vendor]
    if args.providers:
        unknown = set(args.providers) - set(providers)
        if unknown:
            parser.error(f"unsupported provider(s): {sorted(unknown)}")
        providers = tuple(item for item in providers if item == "cpu" or item in args.providers)
    root = Path(__file__).resolve().parent
    webgpu_python = root / ".venv_bench_webgpu" / (
        "Scripts/python.exe" if system == "Windows" else "bin/python"
    )
    adapters = []
    if webgpu_python.is_file():
        probe = subprocess.run(
            [str(webgpu_python), str(root / "probe_webgpu_adapters.py"), "--json"],
            capture_output=True, text=True, check=False,
        )
        if probe.returncode == 0:
            try:
                adapters = json.loads(probe.stdout)
            except json.JSONDecodeError:
                print("WebGPU adapter probe returned non-JSON output", file=sys.stderr)
        else:
            print(f"WebGPU adapter probe failed: {probe.stderr.strip()}", file=sys.stderr)
    if adapters and "webgpu" in providers:
        if args.webgpu_device_index >= len(adapters) or args.webgpu_device_index < 0:
            parser.error("WebGPU device index outside discovered adapters")
        chosen = adapters[args.webgpu_device_index]
        if args.vendor not in chosen["vendor"].lower():
            parser.error(f"WebGPU index {args.webgpu_device_index} is {chosen['vendor']} "
                         f"{chosen['metadata'].get('Description')}; expected {args.vendor}")
    if system == "Windows" and adapters and "directml" in providers:
        matching = [item for item in adapters if
                    item["metadata"].get("DxgiAdapterNumber") == str(args.directml_device_index)]
        if matching and args.vendor not in matching[0]["vendor"].lower():
            parser.error(f"DirectML/DXGI index {args.directml_device_index} is "
                         f"{matching[0]['vendor']} {matching[0]['metadata'].get('Description')}; "
                         f"expected {args.vendor}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("--output-dir must be empty; preserve existing raw results")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "platform": system,
        "vendor": args.vendor,
        "operator_device_name": args.device_name,
        "providers": providers,
        "profiles": args.profiles,
        "batches": args.batches,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "sets": args.sets,
        "detected_webgpu_adapters": adapters,
        "results": [],
    }
    for profile in args.profiles:
        for batch in args.batches:
            for provider in providers:
                executable = root / f".venv_bench_{provider}" / (
                    "Scripts/python.exe" if system == "Windows" else "bin/python"
                )
                output = args.output_dir / f"{profile}-b{batch}-{provider}.json"
                if not executable.is_file():
                    record = {
                        "status": "skipped", "provider_requested": provider,
                        "profile": profile, "batch_size": batch,
                        "error": f"missing virtual environment: {executable}",
                    }
                    output.write_text(json.dumps(record, indent=2), encoding="utf-8")
                    print(f"SKIP {output}: {record['error']}", flush=True)
                else:
                    cmd = [str(executable), str(root / "benchmark_nvidia_ep.py"),
                           "--provider", provider, "--profile", profile,
                           "--batch-size", str(batch), "--output", str(output),
                           "--gpu-index", str(args.gpu_index),
                           "--directml-device-index", str(args.directml_device_index),
                           "--webgpu-device-index", str(args.webgpu_device_index),
                           "--openvino-device", args.openvino_device,
                           "--target-vendor", "cpu" if provider == "cpu" else args.vendor,
                           "--warmup", str(args.warmup),
                           "--iterations", str(args.iterations),
                           "--sets", str(args.sets)]
                    if provider != "cpu":
                        cmd.extend(["--device-name", args.device_name])
                    if args.vendor == "nvidia" and provider in ("directml", "webgpu"):
                        cmd.append("--track-nvidia-memory")
                    result = subprocess.run(cmd, check=False)
                    if not output.is_file():
                        record = {
                            "status": "failed", "provider_requested": provider,
                            "profile": profile, "batch_size": batch,
                            "error": f"benchmark process exited {result.returncode} before writing JSON",
                        }
                        output.write_text(json.dumps(record, indent=2), encoding="utf-8")
                    print(f"CASE {output}: exit={result.returncode}", flush=True)
                manifest["results"].append(str(output))
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}")
    summary = [sys.executable, str(root / "summarize_benchmark_matrix.py"),
               str(args.output_dir), "--output", str(args.output_dir / "summary.json")]
    return subprocess.run(summary, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
