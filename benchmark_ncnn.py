"""Measure DBV4 runtime providers and save comparable probability arrays.

Run this script in the virtual environment for each provider. A separate
process per provider keeps ONNX Runtime distributions isolated.
"""

import argparse
import json
import os
import platform
from pathlib import Path
import statistics
import subprocess
import threading
import time
import importlib.metadata
import hashlib

import numpy as np
from PIL import Image

from embed_tags_universal import load_runtime_model


def gpu_memory_mib(index):
    if index is None:
        return None
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return float(result.strip().splitlines()[0])
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return None


class ResourceMonitor:
    """Sample Linux process RAM and AMD VRAM/GTT while inference runs."""

    def __init__(self, device=None):
        candidates = sorted(Path('/sys/class/drm').glob('card[0-9]*/device'))
        self.device = device or next(
            (path for path in candidates if (path / 'mem_info_gtt_used').is_file()), None)
        self.samples = []
        self.tegra_load = Path('/sys/devices/gpu.0/load')
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def snapshot(self):
        result = {'time': time.perf_counter()}
        try:
            for line in Path('/proc/self/status').read_text().splitlines():
                if line.startswith('VmRSS:'):
                    result['rss_mib'] = int(line.split()[1]) / 1024
        except OSError:
            pass
        if self.device:
            for filename, key, divisor in (
                ('mem_info_vram_used', 'vram_mib', 1048576),
                ('mem_info_gtt_used', 'gtt_mib', 1048576),
                ('gpu_busy_percent', 'gpu_busy_percent', 1),
            ):
                try:
                    result[key] = int((self.device / filename).read_text()) / divisor
                except (OSError, ValueError):
                    pass
        elif self.tegra_load.is_file():
            try:
                # Tegra's nvgpu load counter is in per-mille units.
                result['gpu_busy_percent'] = int(self.tegra_load.read_text()) / 10
            except (OSError, ValueError):
                pass
        try:
            memory = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            result['system_available_mib'] = int(memory['MemAvailable'].split()[0]) / 1024
            result['system_swap_free_mib'] = int(memory['SwapFree'].split()[0]) / 1024
        except (OSError, KeyError, ValueError):
            pass
        return result

    def _run(self):
        while not self.stop.wait(0.2):
            self.samples.append(self.snapshot())

    def __enter__(self):
        self.samples.append(self.snapshot())
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()
        self.samples.append(self.snapshot())

    def summary(self, measured_start):
        result = {'drm_device': str(self.device) if self.device else None,
                  'tegra_load_counter': str(self.tegra_load) if self.tegra_load.is_file() else None,
                  'sample_interval_seconds': 0.2,
                  'baseline': self.samples[0]}
        for key in ('rss_mib', 'vram_mib', 'gtt_mib'):
            values = [sample[key] for sample in self.samples if key in sample]
            result['peak_' + key] = max(values) if values else None
        busy = [sample['gpu_busy_percent'] for sample in self.samples
                if sample['time'] >= measured_start and 'gpu_busy_percent' in sample]
        result['measured_gpu_busy_percent_mean'] = statistics.mean(busy) if busy else None
        result['measured_gpu_busy_percent_peak'] = max(busy) if busy else None
        result['gpu_counters_scope'] = 'whole device, including other processes'
        for key in ('system_available_mib', 'system_swap_free_mib'):
            values = [s[key] for s in self.samples if key in s]
            result['minimum_' + key] = min(values) if values else None
        return result


def compare(reference, probabilities, metadata):
    expected = np.asarray(reference["probabilities"], dtype=np.float32)
    if expected.shape != probabilities.shape:
        raise ValueError(f"基準との出力形状が違います: {expected.shape} != {probabilities.shape}")
    if reference["metadata_version"] != metadata.metadata_version:
        raise ValueError("基準とDBV4 metadata versionが異なります")
    difference = np.abs(expected - probabilities)
    reference_tags = set(reference["selected_tags"])
    actual_tags = set(metadata.decode_tags(probabilities))
    ratings = metadata.get_rating_scores(probabilities)
    return {
        "max_absolute_difference": float(difference.max()),
        "mean_absolute_difference": float(difference.mean()),
        "selected_tag_symmetric_difference": sorted(reference_tags ^ actual_tags),
        "rating_absolute_difference": {
            name: abs(ratings[name] - reference["rating_scores"][name])
            for name in ratings
        },
    }


def package_versions():
    versions = {}
    for name in ('numpy', 'pillow', 'ncnn', 'onnxruntime', 'onnxruntime-gpu',
                 'onnxruntime-openvino', 'onnxruntime-ep-webgpu', 'timm', 'pnnx'):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def benchmark_image():
    y, x = np.indices((480, 640), dtype=np.uint16)
    pixels = np.stack(((x * 7 + y * 11) % 256, (x * 13 + y * 3) % 256,
                       (x * 5 + y * 17) % 256), axis=2).astype(np.uint8)
    return Image.fromarray(pixels, "RGB")


def run(args, monitor):
    image = benchmark_image()
    smi_index = getattr(args, 'nvidia_smi_index', None)
    if smi_index is None and args.provider in ('cuda', 'tensorrt'):
        smi_index = args.gpu_index
    before_memory = gpu_memory_mib(smi_index)
    start = time.perf_counter()
    runtime = load_runtime_model(
        args.provider != "cpu", args.profile, provider=args.provider,
        gpu_index=args.gpu_index, ncnn_precision=args.ncnn_precision,
        openvino_device=args.openvino_device,
        model_file=args.model_file,
        ncnn_part_size_mib=args.ncnn_part_size_mib,
    )
    load_seconds = time.perf_counter() - start
    print(f'[benchmark] Model initialization and startup probe: {load_seconds:.2f}s', flush=True)
    loaded_memory = gpu_memory_mib(smi_index)
    loaded_resources = monitor.snapshot()
    tensor = runtime.preprocessor(image)
    batch = np.repeat(tensor[None, ...], args.batch_size, axis=0)
    # The ncnn graph is batch one; its adapter executes a logical batch
    # sequentially so the default application batch can still be compared.
    for step in range(args.warmup):
        runtime.predict_preprocessed(batch)
        print(f'[benchmark] Warm-up {step + 1}/{args.warmup}', flush=True)
    warm_memory = gpu_memory_mib(smi_index)
    warmed_resources = monitor.snapshot()
    durations = []
    output = None
    measured_start = time.perf_counter()
    for step in range(args.iterations):
        start = time.perf_counter()
        output = runtime.predict_preprocessed(batch)
        durations.append((time.perf_counter() - start) * 1000 / args.batch_size)
        print(f'[benchmark] Iteration {step + 1}/{args.iterations}: '
              f'{durations[-1]:.2f} ms/img', flush=True)
    probabilities = np.asarray(output[0], dtype=np.float32)
    if probabilities.size != runtime.metadata.label_count:
        raise ValueError("DBV4ラベル数が一致しません")
    result = {
        "provider": args.provider,
        "precision": args.ncnn_precision if args.provider == "ncnn" else None,
        "ncnn_part_size_mib": getattr(runtime, 'part_size_mib', args.ncnn_part_size_mib) if args.provider == 'ncnn' else None,
        "profile": args.profile,
        "gpu_index": args.gpu_index,
        "openvino_device": args.openvino_device,
        "nvidia_smi_index": smi_index,
        "platform": platform.platform(),
        "gpu_environment": {name: os.environ[name] for name in (
            'RADV_DEBUG', 'RADV_PERFTEST', 'VK_DRIVER_FILES') if name in os.environ},
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "first_measured_ms_per_image": durations[0],
        "median_ms_per_image": statistics.median(durations),
        "durations_ms_per_image": durations,
        "resources_loaded": loaded_resources,
        "resources_after_warmup": warmed_resources,
        "gpu_memory_mib_before": before_memory,
        "gpu_memory_mib_loaded": loaded_memory,
        "gpu_memory_mib_after_warmup": warm_memory,
        "metadata_version": runtime.metadata.metadata_version,
        "label_count": runtime.metadata.label_count,
        "rating_scores": runtime.metadata.get_rating_scores(probabilities),
        "selected_tags": runtime.metadata.decode_tags(probabilities),
        "probabilities": probabilities.tolist(),
        "package_versions": package_versions(),
        "input_sha256": hashlib.sha256(tensor.tobytes()).hexdigest(),
        "ncnn_device": getattr(runtime, 'device_name', None),
        "ncnn_options": getattr(runtime, 'options', None),
        "ncnn_binding": getattr(runtime, 'binding_info', None),
        "active_providers": (runtime.session.get_providers()
                             if hasattr(runtime, 'session') else ['ncnn Vulkan']),
    }
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        if reference.get('input_sha256') != result['input_sha256']:
            raise ValueError('基準と前処理済み入力が異なります')
        result["comparison"] = compare(reference, probabilities, runtime.metadata)
    return result, measured_start


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt", "intel",
                                               "migraphx", "rocm", "directml", "webgpu", "ncnn"), required=True)
    parser.add_argument("--profile", default="ultra")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--openvino-device", help="Explicit OpenVINO GPU.N device")
    parser.add_argument("--nvidia-smi-index", type=int,
                        help="NVIDIA memory counter index; Vulkan numbering can differ")
    parser.add_argument("--ncnn-precision", choices=("fp32", "fp16-storage",
                                                      "fp16-packed", "fp16-arithmetic"),
                        default="fp32")
    parser.add_argument("--model-file")
    parser.add_argument("--ncnn-part-size-mib", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--drm-device", type=Path, help="AMD counters, e.g. /sys/class/drm/card1/device")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.warmup, args.iterations) < 1:
        parser.error("batch-size, warmup, iterations must be positive")
    with ResourceMonitor(args.drm_device) as monitor:
        result, measured_start = run(args, monitor)
    result['resources'] = monitor.summary(measured_start)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"{args.output}: {result['median_ms_per_image']:.2f} ms/img")


if __name__ == "__main__":
    main()
