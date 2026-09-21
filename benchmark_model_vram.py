"""Measure one model profile's GPU memory use without reading user image folders."""

import argparse
import gc
import json
import subprocess
import threading
import time
from typing import List

from PIL import Image

from embed_tags_universal import load_runtime_model


def gpu_used_mib() -> int:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        encoding="utf-8",
    )
    return int(output.strip().splitlines()[0])


def measure(profile: str, batch_size: int, iterations: int) -> dict:
    samples: List[int] = []
    stop_event = threading.Event()
    baseline = gpu_used_mib()

    def poll() -> None:
        while not stop_event.wait(0.1):
            try:
                samples.append(gpu_used_mib())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    poller = threading.Thread(target=poll, daemon=True)
    poller.start()
    started = time.perf_counter()
    runtime = load_runtime_model(True, profile)
    images = [Image.new("RGB", (640, 480), (127, 63, 191)) for _ in range(batch_size)]
    inference_times = []
    try:
        for _ in range(iterations):
            iteration_started = time.perf_counter()
            runtime.predict_images(images)
            inference_times.append(time.perf_counter() - iteration_started)
        resident = gpu_used_mib()
    finally:
        stop_event.set()
        poller.join(timeout=2.0)
        del runtime
        gc.collect()

    peak = max(samples + [baseline, resident])
    return {
        "profile": profile,
        "batch_size": batch_size,
        "iterations": iterations,
        "baseline_mib": baseline,
        "resident_mib": resident,
        "peak_mib": peak,
        "resident_increment_mib": max(0, resident - baseline),
        "peak_increment_mib": max(0, peak - baseline),
        "average_ms_per_image": (
            sum(inference_times) * 1000.0 / (len(inference_times) * batch_size)
        ),
        "total_seconds": time.perf_counter() - started,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "モデルをDirectMLでロードし、合成画像のウォームアップ／推論中に"
            "nvidia-smiでVRAMを測定します。"
        )
    )
    parser.add_argument("profile", help="測定するmodel profile名")
    parser.add_argument("--batch-size", type=int, default=4, help="batch size（既定: 4）")
    parser.add_argument("--iterations", type=int, default=3, help="推論回数（既定: 3）")
    args = parser.parse_args()
    if args.batch_size < 1 or args.iterations < 1:
        parser.error("--batch-sizeと--iterationsは1以上を指定してください。")
    print(json.dumps(measure(args.profile, args.batch_size, args.iterations), ensure_ascii=False))


if __name__ == "__main__":
    main()
