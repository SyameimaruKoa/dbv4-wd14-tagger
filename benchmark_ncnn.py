"""Measure DBV4 runtime providers and save comparable probability arrays.

Run this script in the virtual environment for each provider. A separate
process per provider keeps ONNX Runtime distributions isolated.
"""

import argparse
import json
import platform
from pathlib import Path
import statistics
import subprocess
import time

import numpy as np
from PIL import Image

from embed_tags_universal import load_runtime_model


def gpu_memory_mib(index):
    try:
        result = subprocess.check_output(
            ["nvidia-smi", "-i", str(index), "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"], text=True, timeout=5,
            stderr=subprocess.DEVNULL,
        )
        return float(result.strip().splitlines()[0])
    except (FileNotFoundError, ValueError, subprocess.SubprocessError):
        return None


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


def run(args):
    y, x = np.indices((480, 640), dtype=np.uint16)
    pixels = np.stack(((x * 7 + y * 11) % 256, (x * 13 + y * 3) % 256,
                       (x * 5 + y * 17) % 256), axis=2).astype(np.uint8)
    image = Image.fromarray(pixels, "RGB")
    before_memory = gpu_memory_mib(args.gpu_index)
    start = time.perf_counter()
    runtime = load_runtime_model(
        args.provider != "cpu", args.profile, provider=args.provider,
        gpu_index=args.gpu_index, ncnn_precision=args.ncnn_precision,
        model_file=args.model_file,
    )
    load_seconds = time.perf_counter() - start
    loaded_memory = gpu_memory_mib(args.gpu_index)
    tensor = runtime.preprocessor(image)
    batch = np.repeat(tensor[None, ...], args.batch_size, axis=0)
    # The ncnn graph is batch one; its adapter executes a logical batch
    # sequentially so the default application batch can still be compared.
    for _ in range(args.warmup):
        runtime.predict_preprocessed(batch)
    warm_memory = gpu_memory_mib(args.gpu_index)
    durations = []
    output = None
    for _ in range(args.iterations):
        start = time.perf_counter()
        output = runtime.predict_preprocessed(batch)
        durations.append((time.perf_counter() - start) * 1000 / args.batch_size)
    probabilities = np.asarray(output[0], dtype=np.float32)
    if probabilities.size != runtime.metadata.label_count:
        raise ValueError("DBV4ラベル数が一致しません")
    result = {
        "provider": args.provider,
        "precision": args.ncnn_precision if args.provider == "ncnn" else None,
        "profile": args.profile,
        "platform": platform.platform(),
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "load_seconds": load_seconds,
        "first_measured_ms_per_image": durations[0],
        "median_ms_per_image": statistics.median(durations),
        "gpu_memory_mib_before": before_memory,
        "gpu_memory_mib_loaded": loaded_memory,
        "gpu_memory_mib_after_warmup": warm_memory,
        "metadata_version": runtime.metadata.metadata_version,
        "label_count": runtime.metadata.label_count,
        "rating_scores": runtime.metadata.get_rating_scores(probabilities),
        "selected_tags": runtime.metadata.decode_tags(probabilities),
        "probabilities": probabilities.tolist(),
    }
    if args.reference:
        reference = json.loads(args.reference.read_text(encoding="utf-8"))
        result["comparison"] = compare(reference, probabilities, runtime.metadata)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt", "intel",
                                               "migraphx", "rocm", "directml", "webgpu", "ncnn"), required=True)
    parser.add_argument("--profile", default="ultra")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--ncnn-precision", choices=("fp32", "fp16-storage",
                                                      "fp16-packed", "fp16-arithmetic"),
                        default="fp32")
    parser.add_argument("--model-file")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if min(args.batch_size, args.warmup, args.iterations) < 1:
        parser.error("batch-size, warmup, iterations must be positive")
    result = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(f"{args.output}: {result['median_ms_per_image']:.2f} ms/img")


if __name__ == "__main__":
    main()
