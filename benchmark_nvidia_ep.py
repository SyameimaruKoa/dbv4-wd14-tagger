"""Run one repeatable GPU provider benchmark and write its raw JSON result.

Run each provider in its own virtual environment; see WEBGPU_BENCHMARK_WINDOWS.md
and WEBGPU_BENCHMARK_LINUX.md. This script does not change user images.
"""

import argparse
import json
import platform
import statistics
import subprocess
import threading
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import onnxruntime as ort
import psutil
from PIL import Image

from dbv4 import (
    DBV4Metadata,
    DBV4Preprocessor,
    MODEL_PROFILES,
    adapt_input_layout,
    select_output_name,
)
from gpu_runtime import (
    prepare_tensorrt, prepare_cuda,
    prepare_openvino, configure_directml, configure_webgpu, executed_providers,
)


def percentile(values, fraction):
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def nvidia_memory_mib(index):
    output = subprocess.check_output(
        ["nvidia-smi", "-i", str(index), "--query-gpu=memory.used",
         "--format=csv,noheader,nounits"], text=True, timeout=5,
    )
    return float(output.strip().splitlines()[0])



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt", "webgpu", "intel", "directml", "migraphx"), required=True)
    parser.add_argument("--openvino-device", default="GPU.0")
    parser.add_argument("--tensorrt-lib-dir", type=Path,
                        help="TensorRT 10 bin directory; auto-detected on Windows when omitted")
    parser.add_argument("--target-vendor", choices=("cpu", "intel", "nvidia", "amd"))
    parser.add_argument("--device-name", help="Operator-confirmed adapter name for DirectML/WebGPU/MIGraphX")
    parser.add_argument("--profile", choices=tuple(MODEL_PROFILES), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--directml-device-index", type=int, default=0)
    parser.add_argument("--track-nvidia-memory", action="store_true")
    parser.add_argument("--webgpu-device-index", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--sets", type=int, default=3)
    args = parser.parse_args()
    if min(args.batch_size, args.warmup, args.iterations, args.sets) < 1:
        parser.error("batch-size, warmup, iterations and sets must be positive")
    profile = MODEL_PROFILES[args.profile]
    if not profile.get("available", True):
        parser.error(str(profile.get("unavailable_reason", "profile unavailable")))

    track_nvidia = args.provider in ("cuda", "tensorrt") or args.track_nvidia_memory
    if args.provider == "cpu" and args.target_vendor not in (None, "cpu"):
        parser.error("CPU provider requires target vendor cpu")
    if args.provider == "intel" and args.target_vendor not in (None, "intel"):
        parser.error("OpenVINO Intel provider requires target vendor intel")
    if args.provider in ("directml", "webgpu", "migraphx") and args.target_vendor and not args.device_name:
        parser.error("this provider requires --device-name when --target-vendor is set")
    if args.provider in ("directml", "webgpu", "migraphx") and args.target_vendor:
        if args.target_vendor not in args.device_name.lower():
            parser.error("--device-name must include the target vendor")
    if args.provider in ("cuda", "tensorrt") and args.target_vendor not in (None, "nvidia"):
        parser.error("CUDA/TensorRT require target vendor nvidia")
    try:
        baseline_vram = nvidia_memory_mib(args.gpu_index) if track_nvidia else None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error(f"nvidia-smi cannot read GPU {args.gpu_index}: {exc}")
    baseline_rss = psutil.Process().memory_info().rss / 1048576
    samples_vram = [baseline_vram] if track_nvidia else []
    samples_rss = [baseline_rss]
    stop = threading.Event()

    def monitor():
        process = psutil.Process()
        while not stop.wait(0.1):
            try:
                if track_nvidia:
                    samples_vram.append(nvidia_memory_mib(args.gpu_index))
                samples_rss.append(process.memory_info().rss / 1048576)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    session = None
    profile_stopped = False
    webgpu_hardware = None
    tensorrt_lib_dir = None
    try:
        if args.provider == "tensorrt":
            tensorrt_lib_dir = prepare_tensorrt(args.tensorrt_lib_dir)
        started = time.perf_counter()
        metadata = DBV4Metadata.load(profile, base_dir=str(Path(__file__).resolve().parent))
        metadata_seconds = time.perf_counter() - started
        runtime_handles = []
        if args.provider in ("cuda", "tensorrt"):
            runtime_handles.extend(prepare_cuda())
        openvino_gpu_name = None
        if args.provider == "intel":
            openvino_gpu_name = prepare_openvino(args.openvino_device)
        options = ort.SessionOptions()
        options.enable_profiling = True
        if args.provider in ("cpu", "cuda", "tensorrt", "intel", "directml", "migraphx"):
            if args.provider == "directml":
                configure_directml(options)
            available = ort.get_available_providers()
            names = {
                "cuda": "CUDAExecutionProvider",
                "tensorrt": "TensorrtExecutionProvider",
                "intel": "OpenVINOExecutionProvider",
                "directml": "DmlExecutionProvider",
                "migraphx": "MIGraphXExecutionProvider",
                "cpu": "CPUExecutionProvider",
            }
            expected_provider = names[args.provider]
            if expected_provider not in available:
                raise RuntimeError(f"{expected_provider} unavailable: {available}")
            if args.provider == "cpu":
                providers = [expected_provider]
            elif args.provider == "cuda":
                providers = [(expected_provider, {"device_id": str(args.gpu_index)}),
                             "CPUExecutionProvider"]
            elif args.provider == "tensorrt":
                providers = [(expected_provider, {"device_id": str(args.gpu_index)}),
                             ("CUDAExecutionProvider", {"device_id": str(args.gpu_index)}),
                             "CPUExecutionProvider"]
            elif args.provider == "intel":
                providers = [(expected_provider, {"device_type": args.openvino_device}),
                             "CPUExecutionProvider"]
            elif args.provider == "migraphx":
                providers = [(expected_provider, {"device_id": str(args.gpu_index)}),
                             "CPUExecutionProvider"]
            else:
                providers = [(expected_provider, {"device_id": str(args.directml_device_index)}),
                             "CPUExecutionProvider"]
            started = time.perf_counter()
            session = ort.InferenceSession(metadata.model_path, sess_options=options,
                                           providers=providers)
        else:
            webgpu_hardware = configure_webgpu(
                options, args.webgpu_device_index, args.target_vendor)
            started = time.perf_counter()
            session = ort.InferenceSession(metadata.model_path, sess_options=options)
            expected_provider = "WebGpuExecutionProvider"
        session_seconds = time.perf_counter() - started
        session.disable_fallback()
        active = session.get_providers()
        if expected_provider not in active:
            raise RuntimeError(f"requested {expected_provider}, active providers are {active}")

        rng = np.random.default_rng(20260922)
        pixels = rng.integers(0, 256, (480, 640, 3), dtype=np.uint8)
        image = Image.fromarray(pixels, "RGB")
        preprocessor = DBV4Preprocessor.from_metadata(metadata)
        input_shape = session.get_inputs()[0].shape
        batch = np.stack([preprocessor(image) for _ in range(args.batch_size)]).astype(np.float32)
        batch = adapt_input_layout(batch, input_shape)
        feed = {session.get_inputs()[0].name: batch}
        output_name = select_output_name(session.get_outputs(), metadata.label_count)
        for _ in range(args.warmup):
            session.run([output_name], feed)
        timings = []
        result_shape = None
        for _ in range(args.sets):
            one_set = []
            for _ in range(args.iterations):
                started = time.perf_counter()
                output = session.run([output_name], feed)[0]
                one_set.append((time.perf_counter() - started) * 1000 / args.batch_size)
                result_shape = list(np.asarray(output).shape)
            timings.append(one_set)
        if result_shape != [args.batch_size, metadata.label_count]:
            raise RuntimeError(f"unexpected output shape: {result_shape}")
        stop.set()
        watcher.join(timeout=2)
        peak_vram = max(samples_vram + [nvidia_memory_mib(args.gpu_index)]) if track_nvidia else None
        profile_path = Path(session.end_profiling())
        profile_stopped = True
        try:
            executed = executed_providers(profile_path)
        finally:
            profile_path.unlink(missing_ok=True)
        if expected_provider not in executed:
            raise RuntimeError(f"{expected_provider} did not execute a profiled node: {executed}")
        record = {
            "status": "ok", "provider_requested": args.provider,
            "provider_active": active, "profile": args.profile,
            "repo_id": metadata.repo_id, "metadata_version": metadata.metadata_version,
            "model_path": metadata.model_path, "batch_size": args.batch_size,
            "input_shape": input_shape, "output_shape": result_shape,
            "seed": 20260922, "image_size": [640, 480],
            "warmup": args.warmup, "iterations_per_set": args.iterations,
            "sets": args.sets, "raw_ms_per_image": timings,
            "set_median_ms": [statistics.median(values) for values in timings],
            "set_p95_ms": [percentile(values, 0.95) for values in timings],
            "median_ms_per_image": statistics.median(
                [statistics.median(values) for values in timings]
            ),
            "baseline_vram_mib": baseline_vram, "peak_vram_mib": peak_vram,
            "vram_increment_mib": peak_vram - baseline_vram if track_nvidia else None,
            "baseline_rss_mib": baseline_rss, "peak_rss_mib": max(samples_rss),
            "metadata_seconds": metadata_seconds, "session_seconds": session_seconds,
            "platform": platform.platform(), "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "webgpu_plugin": version("onnxruntime-ep-webgpu") if args.provider == "webgpu" else None,
            "gpu_index": args.gpu_index,
            "directml_device_index": args.directml_device_index if args.provider == "directml" else None,
            "webgpu_device_index": args.webgpu_device_index if args.provider == "webgpu" else None,
            "nvidia_gpu_memory_observed": peak_vram - baseline_vram >= 16 if track_nvidia else None,
            "executed_node_providers": executed,
            "openvino_device": args.openvino_device if args.provider == "intel" else None,
            "openvino_gpu_name": openvino_gpu_name,
            "webgpu_hardware": webgpu_hardware,
            "tensorrt_lib_dir": str(tensorrt_lib_dir) if tensorrt_lib_dir else None,
            "target_vendor": args.target_vendor,
            "device_name": openvino_gpu_name if args.provider == "intel" else args.device_name,
        }
    except Exception as exc:
        record = {
            "status": "failed", "provider_requested": args.provider,
            "profile": args.profile, "batch_size": args.batch_size,
            "target_vendor": args.target_vendor, "device_name": args.device_name,
            "error": f"{type(exc).__name__}: {exc}",
            "platform": platform.platform(), "onnxruntime": ort.__version__,
        }
    finally:
        stop.set()
        watcher.join(timeout=2)
        if session is not None and not profile_stopped:
            try:
                Path(session.end_profiling()).unlink(missing_ok=True)
            except (OSError, RuntimeError):
                pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if record["status"] == "ok":
        print(f"Saved {args.output}: {record['median_ms_per_image']:.2f} ms/image")
    else:
        raise RuntimeError(f"Saved failure details to {args.output}: {record['error']}")


if __name__ == "__main__":
    main()
