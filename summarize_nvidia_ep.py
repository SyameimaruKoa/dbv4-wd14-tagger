"""Summarize CUDA/WebGPU benchmark JSON files into a small comparison JSON."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.input_dir.glob("*.json")):
        if path.resolve() == args.output.resolve():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "profile" in row and "provider_requested" in row:
            rows.append(row)
    pairs = {}
    for row in rows:
        key = (row["profile"], row["batch_size"])
        pairs.setdefault(key, {})[row["provider_requested"]] = row
    summary = {"comparisons": [], "incomplete": []}
    for (profile, batch), pair in sorted(pairs.items()):
        cuda = pair.get("cuda")
        webgpu = pair.get("webgpu")
        if not cuda or not webgpu or cuda["status"] != "ok" or webgpu["status"] != "ok":
            summary["incomplete"].append({
                "profile": profile, "batch_size": batch,
                "cuda": cuda.get("error", cuda.get("status")) if cuda else "missing",
                "webgpu": webgpu.get("error", webgpu.get("status")) if webgpu else "missing",
            })
            continue
        comparable = (
            "repo_id", "metadata_version", "seed", "image_size", "warmup",
            "iterations_per_set", "sets", "input_shape", "output_shape",
        )
        mismatched = [name for name in comparable if cuda.get(name) != webgpu.get(name)]
        if mismatched or not webgpu.get("nvidia_gpu_memory_observed"):
            summary["incomplete"].append({
                "profile": profile, "batch_size": batch,
                "reason": (
                    f"mismatched inputs: {', '.join(mismatched)}" if mismatched
                    else "WebGPU NVIDIA VRAM increment not observed"
                ),
            })
            continue
        cuda_ms = cuda["median_ms_per_image"]
        webgpu_ms = webgpu["median_ms_per_image"]
        summary["comparisons"].append({
            "profile": profile, "batch_size": batch,
            "cuda_ms_per_image": round(cuda_ms, 3),
            "webgpu_ms_per_image": round(webgpu_ms, 3),
            "webgpu_over_cuda_ratio": round(webgpu_ms / cuda_ms, 3),
            "webgpu_slowdown_pct": round((webgpu_ms / cuda_ms - 1) * 100, 1),
            "cuda_vram_increment_mib": cuda["vram_increment_mib"],
            "webgpu_vram_increment_mib": webgpu["vram_increment_mib"],
            "webgpu_nvidia_memory_observed": webgpu["nvidia_gpu_memory_observed"],
            "cuda_active": cuda["provider_active"],
            "webgpu_active": webgpu["provider_active"],
            "cuda_version": cuda["onnxruntime"],
            "webgpu_version": webgpu["onnxruntime"],
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}: {len(summary['comparisons'])} pairs, "
          f"{len(summary['incomplete'])} incomplete")


if __name__ == "__main__":
    main()
