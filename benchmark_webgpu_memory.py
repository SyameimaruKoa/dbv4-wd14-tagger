"""Measure WebGPU model memory and steady-state inference on Linux AMD GPUs.

Each profile runs in a fresh child process. The parent samples process RSS and
amdgpu's dedicated VRAM/GTT counters, including model initialization and warmup.
"""

import argparse
import datetime
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image

# Keep multi-gigabyte models off systems whose default ~/.cache is tmpfs.
os.environ.setdefault(
    "HF_HOME", str(Path.home() / ".local/share/huggingface")
)
os.environ.setdefault(
    "HF_HUB_CACHE", str(Path.home() / ".local/share/huggingface/hub")
)
os.environ.setdefault(
    "HF_XET_CACHE", str(Path.home() / ".local/share/huggingface/xet")
)

from dbv4 import MODEL_PROFILES


GPU_DEVICE = Path("/sys/class/drm/card1/device")
MIB = 1024 * 1024


def read_int(path):
    try:
        return int(Path(path).read_text().strip())
    except (OSError, ValueError):
        return None


def ram_total_mib():
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            return int(line.split()[1]) / 1024
    raise RuntimeError("MemTotal is unavailable")


def sample(pid=None):
    data = {
        "vram_used_mib": (read_int(GPU_DEVICE / "mem_info_vram_used") or 0) / MIB,
        "gtt_used_mib": (read_int(GPU_DEVICE / "mem_info_gtt_used") or 0) / MIB,
    }
    if pid is not None:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    data["rss_mib"] = int(line.split()[1]) / 1024
                elif line.startswith("VmSwap:"):
                    data["process_swap_mib"] = int(line.split()[1]) / 1024
        except OSError:
            pass
    try:
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in ("SwapTotal", "SwapFree", "MemAvailable"):
                memory[key] = int(value.split()[0]) / 1024
        data["system_swap_used_mib"] = memory["SwapTotal"] - memory["SwapFree"]
        data["system_mem_available_mib"] = memory["MemAvailable"]
    except (OSError, KeyError):
        pass
    return data


def worker(profile, iterations, output):
    from embed_tags_universal import load_runtime_model

    started = time.perf_counter()
    runtime = load_runtime_model(True, profile, use_webgpu=True)
    load_seconds = time.perf_counter() - started
    after_load = sample(os.getpid())
    image = Image.new("RGB", (640, 480), (127, 63, 191))
    images = [image]
    prepared = runtime.preprocess_batch(images)
    feed = {runtime.input_name: prepared}
    for _ in range(3):
        runtime.session.run([runtime.output_name], feed)
    raw_times = []
    full_times = []
    for _ in range(iterations):
        started = time.perf_counter()
        runtime.session.run([runtime.output_name], feed)
        raw_times.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        runtime.predict_images(images)
        full_times.append((time.perf_counter() - started) * 1000)
    result = {
        "profile": profile,
        "provider": runtime.session.get_providers(),
        "model_repo": runtime.metadata.repo_id,
        "input_shape": runtime.input_shape,
        "batch_size": 1,
        "warmup_iterations": 3,
        "measured_iterations": iterations,
        "load_seconds": round(load_seconds, 3),
        "after_load": after_load,
        "inference_only_ms": [round(value, 3) for value in raw_times],
        "end_to_end_ms": [round(value, 3) for value in full_times],
        "inference_only_median_ms": round(statistics.median(raw_times), 3),
        "end_to_end_median_ms": round(statistics.median(full_times), 3),
    }
    Path(output).write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")


def measure(profile, iterations):
    before = sample()
    with tempfile.TemporaryDirectory(prefix="dbv4-webgpu-") as directory:
        result_path = Path(directory) / "result.json"
        log_path = Path(directory) / "worker.log"
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [sys.executable, __file__, "--worker", profile,
                 "--iterations", str(iterations), "--output", str(result_path)],
                stdout=log, stderr=subprocess.STDOUT,
            )
            peak = sample(process.pid)
            while process.poll() is None:
                current = sample(process.pid)
                for key, value in current.items():
                    if key == "system_mem_available_mib":
                        peak[key] = min(peak.get(key, value), value)
                    else:
                        peak[key] = max(peak.get(key, 0), value)
                time.sleep(0.05)
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if process.returncode != 0 or not result_path.exists():
            return {
                "profile": profile, "status": "failed",
                "exit_code": process.returncode,
                "error_tail": log_text[-3000:],
                "peak": peak,
            }
        result = json.loads(result_path.read_text(encoding="utf-8"))
        result.update({
            "status": "ok", "before": before, "peak": peak,
            "vram_peak_increment_mib": round(peak["vram_used_mib"] - before["vram_used_mib"], 1),
            "gtt_peak_increment_mib": round(peak["gtt_used_mib"] - before["gtt_used_mib"], 1),
            "vram_peak_pct_total": round(
                peak["vram_used_mib"] / (read_int(GPU_DEVICE / "mem_info_vram_total") / MIB) * 100, 1
            ),
            "gtt_peak_pct_total": round(
                peak["gtt_used_mib"] / (read_int(GPU_DEVICE / "mem_info_gtt_total") / MIB) * 100, 1
            ),
            "rss_peak_pct_ram": round(peak["rss_mib"] / ram_total_mib() * 100, 1),
        })
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="*", default=[
        "lightweight", "balanced", "high", "ultra", "wd14_v3",
    ])
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", default="benchmarks/amd_barcelo_512mb.json")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        worker(args.worker, args.iterations, args.output)
        return
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    total = read_int(GPU_DEVICE / "mem_info_vram_total")
    if total is None:
        parser.error(f"amdgpu VRAM counters not found at {GPU_DEVICE}")
    report = {
        "gpu_vram_total_mib": round(total / MIB, 1) if total else None,
        "gpu_gtt_total_mib": round((read_int(GPU_DEVICE / "mem_info_gtt_total") or 0) / MIB, 1),
        "system_ram_total_mib": round(ram_total_mib(), 1),
        "environment": {
            "measured_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
            "system": platform.platform(),
            "python": platform.python_version(),
            "onnxruntime": importlib.metadata.version("onnxruntime"),
            "onnxruntime_ep_webgpu": importlib.metadata.version("onnxruntime-ep-webgpu"),
            "gpu_pci_device": (GPU_DEVICE / "device").read_text().strip(),
        },
        "conditions": f"Linux AMD WebGPU; RGB 640x480 solid color (127,63,191); batch=1; warmup=3; {args.iterations} measured iterations; 50ms memory sampling",
        "results": [],
    }
    output = Path(args.output)
    if output.exists():
        prior = json.loads(output.read_text(encoding="utf-8"))
        if prior.get("gpu_vram_total_mib") != report["gpu_vram_total_mib"]:
            parser.error("output file belongs to a different VRAM configuration")
        report["results"] = prior.get("results", [])
    for profile in args.profiles:
        if profile not in MODEL_PROFILES:
            parser.error(f"unknown profile: {profile}")
        print(f"[BENCH] {profile}", flush=True)
        result = measure(profile, args.iterations)
        report["results"] = [item for item in report["results"] if item["profile"] != profile]
        report["results"].append(result)
        print(json.dumps({key: value for key, value in result.items()
                          if key in ("profile", "status", "inference_only_median_ms",
                                     "end_to_end_median_ms", "vram_peak_increment_mib",
                                     "gtt_peak_increment_mib", "error_tail")}, ensure_ascii=False), flush=True)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
