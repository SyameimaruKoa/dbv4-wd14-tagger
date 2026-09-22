"""Run one repeatable CUDA or WebGPU benchmark and write its raw JSON result.

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
    parser.add_argument("--provider", choices=("cuda", "webgpu"), required=True)
    parser.add_argument("--profile", choices=tuple(MODEL_PROFILES), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
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

    try:
        baseline_vram = nvidia_memory_mib(args.gpu_index)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error(f"nvidia-smi cannot read GPU {args.gpu_index}: {exc}")
    baseline_rss = psutil.Process().memory_info().rss / 1048576
    samples_vram = [baseline_vram]
    samples_rss = [baseline_rss]
    stop = threading.Event()

    def monitor():
        process = psutil.Process()
        while not stop.wait(0.1):
            try:
                samples_vram.append(nvidia_memory_mib(args.gpu_index))
                samples_rss.append(process.memory_info().rss / 1048576)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    try:
        started = time.perf_counter()
        metadata = DBV4Metadata.load(profile, base_dir=str(Path(__file__).resolve().parent))
        metadata_seconds = time.perf_counter() - started
        options = ort.SessionOptions()
        if args.provider == "cuda":
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(f"CUDA EP unavailable: {available}")
            providers = [("CUDAExecutionProvider", {"device_id": str(args.gpu_index)}),
                         "CPUExecutionProvider"]
            started = time.perf_counter()
            session = ort.InferenceSession(metadata.model_path, sess_options=options,
                                           providers=providers)
            expected_provider = "CUDAExecutionProvider"
        else:
            import onnxruntime_ep_webgpu as webgpu

            ort.register_execution_provider_library("benchmark_webgpu", webgpu.get_library_path())
            devices = [device for device in ort.get_ep_devices()
                       if device.ep_name == webgpu.get_ep_name()]
            if args.webgpu_device_index >= len(devices):
                raise RuntimeError(f"WebGPU device index unavailable: {len(devices)} devices")
            options.add_provider_for_devices([devices[args.webgpu_device_index]], {})
            started = time.perf_counter()
            session = ort.InferenceSession(metadata.model_path, sess_options=options)
            expected_provider = webgpu.get_ep_name()
        session_seconds = time.perf_counter() - started
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
        peak_vram = max(samples_vram + [nvidia_memory_mib(args.gpu_index)])
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
            "vram_increment_mib": peak_vram - baseline_vram,
            "baseline_rss_mib": baseline_rss, "peak_rss_mib": max(samples_rss),
            "metadata_seconds": metadata_seconds, "session_seconds": session_seconds,
            "platform": platform.platform(), "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "webgpu_plugin": version("onnxruntime-ep-webgpu") if args.provider == "webgpu" else None,
            "gpu_index": args.gpu_index,
            "webgpu_device_index": args.webgpu_device_index if args.provider == "webgpu" else None,
            "nvidia_gpu_memory_observed": peak_vram - baseline_vram >= 16,
        }
    except Exception as exc:
        record = {
            "status": "failed", "provider_requested": args.provider,
            "profile": args.profile, "batch_size": args.batch_size,
            "error": f"{type(exc).__name__}: {exc}",
            "platform": platform.platform(), "onnxruntime": ort.__version__,
        }
    finally:
        stop.set()
        watcher.join(timeout=2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if record["status"] == "ok":
        print(f"Saved {args.output}: {record['median_ms_per_image']:.2f} ms/image")
    else:
        raise RuntimeError(f"Saved failure details to {args.output}: {record['error']}")


if __name__ == "__main__":
    main()
