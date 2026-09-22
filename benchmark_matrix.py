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
    parser.add_argument("--tensorrt-lib-dir", type=Path,
                        help="TensorRT 10 bin directory; auto-detected on Windows")
    parser.add_argument("--profiles", nargs="+",
                        default=["wd14_v3", "balanced"])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--providers", nargs="+",
                        help="Subset of the platform/vendor providers; cpu is always included")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sets", type=int, default=3)
    parser.add_argument("--retry-failed", action="store_true",
                        help="Resume an existing completed matrix and rerun failed cases only")
    parser.add_argument("--retry-provider", action="append",
                        help="With --retry-failed, rerun only this provider (repeatable)")
    args = parser.parse_args()
    from dbv4 import MODEL_PROFILES

    invalid_profiles = [name for name in args.profiles if name not in MODEL_PROFILES]
    if invalid_profiles:
        parser.error(f"unknown profile(s): {invalid_profiles}")
    system = platform.system()
    if system not in MATRIX:
        parser.error(f"unsupported OS: {system}")
    if args.retry_provider and not args.retry_failed:
        parser.error("--retry-provider requires --retry-failed")
    if args.vendor not in args.device_name.lower():
        parser.error("--device-name must include --vendor; confirm the actual adapter")
    if min(*args.batches, args.warmup, args.iterations, args.sets) < 1:
        parser.error("batch and repetition counts must be positive")
    providers = ("cpu",) + MATRIX[system][args.vendor]
    if args.retry_provider and set(args.retry_provider) - set(providers):
        parser.error("--retry-provider must be a provider in this matrix")
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
    existing_manifest = args.output_dir / "manifest.json"
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        if not args.retry_failed:
            parser.error("--output-dir must be empty; use --retry-failed to preserve successes")
        if not existing_manifest.is_file():
            parser.error("cannot resume: manifest.json is missing (run may still be in progress)")
        previous = json.loads(existing_manifest.read_text(encoding="utf-8"))
        expected = {"platform": system, "vendor": args.vendor,
                    "operator_device_name": args.device_name,
                    "providers": list(providers), "profiles": args.profiles,
                    "batches": args.batches, "warmup": args.warmup,
                    "iterations": args.iterations, "sets": args.sets}
        mismatch = [key for key, value in expected.items() if previous.get(key) != value]
        if mismatch:
            parser.error(f"cannot resume: matrix settings differ: {mismatch}")
    elif args.retry_failed:
        parser.error("--retry-failed requires an existing completed matrix")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = previous if args.retry_failed else {
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
                if args.retry_failed:
                    if args.retry_provider and provider not in args.retry_provider:
                        print(f"KEEP {output}: provider not selected for retry", flush=True)
                        continue
                    if output.is_file():
                        prior = json.loads(output.read_text(encoding="utf-8"))
                        if prior.get("status") == "ok":
                            print(f"KEEP {output}: successful result preserved", flush=True)
                            continue
                migraphx_issue = None
                if provider == "migraphx" and executable.is_file():
                    probe = subprocess.run(
                        [str(executable), "-c", "import onnxruntime as ort; "
                         "print(ort.get_available_providers())"],
                        capture_output=True, text=True, check=False,
                    )
                    if probe.returncode != 0:
                        detail = probe.stderr.strip().splitlines()[-1] if probe.stderr.strip() else "probe failed"
                        migraphx_issue = f"MIGraphX ONNX Runtime unavailable: {detail}"
                    elif "MIGraphXExecutionProvider" not in probe.stdout:
                        migraphx_issue = "MIGraphXExecutionProvider unavailable: " + probe.stdout.strip()
                if not executable.is_file() or migraphx_issue:
                    record = {
                        "status": "skipped", "provider_requested": provider,
                        "profile": profile, "batch_size": batch,
                        "error": migraphx_issue or f"missing virtual environment: {executable}",
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
                    if provider == "tensorrt" and args.tensorrt_lib_dir:
                        cmd.extend(["--tensorrt-lib-dir", str(args.tensorrt_lib_dir)])
                    if provider != "cpu":
                        cmd.extend(["--device-name", args.device_name])
                    if args.vendor == "nvidia" and provider in ("directml", "webgpu"):
                        cmd.append("--track-nvidia-memory")
                    case_env = None
                    if provider == "tensorrt" and system == "Linux":
                        import os

                        trt_dir = args.tensorrt_lib_dir
                        if trt_dir is None and os.environ.get("TENSORRT_LIB_DIR"):
                            trt_dir = Path(os.environ["TENSORRT_LIB_DIR"])
                        if trt_dir is None:
                            venv = executable.parent.parent
                            for pattern in ("lib/python*/site-packages/tensorrt*_libs",
                                            "lib/python*/site-packages/tensorrt*_libs/lib"):
                                found = [item for item in venv.glob(pattern)
                                         if (item / "libnvinfer.so.10").is_file()]
                                if len(found) == 1:
                                    trt_dir = found[0]
                                    break
                        case_env = os.environ.copy()
                        library_dirs = []
                        if trt_dir is not None:
                            case_env["TENSORRT_LIB_DIR"] = str(trt_dir.resolve())
                            library_dirs.append(str(trt_dir.resolve()))
                        library_dirs.extend(
                            str(item.resolve()) for item in
                            executable.parent.parent.glob(
                                "lib/python*/site-packages/nvidia/*/lib"
                            ) if item.is_dir()
                        )
                        if library_dirs:
                            case_env["LD_LIBRARY_PATH"] = (
                                os.pathsep.join(library_dirs) + os.pathsep +
                                case_env.get("LD_LIBRARY_PATH", "")
                            )
                    result = subprocess.run(cmd, check=False, env=case_env)
                    if not output.is_file():
                        record = {
                            "status": "failed", "provider_requested": provider,
                            "profile": profile, "batch_size": batch,
                            "error": f"benchmark process exited {result.returncode} before writing JSON",
                        }
                        output.write_text(json.dumps(record, indent=2), encoding="utf-8")
                    print(f"CASE {output}: exit={result.returncode}", flush=True)
                if not args.retry_failed:
                    manifest["results"].append(str(output))
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {manifest_path}")
    summary = [sys.executable, str(root / "summarize_benchmark_matrix.py"),
               str(args.output_dir), "--output", str(args.output_dir / "summary.json")]
    return subprocess.run(summary, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
