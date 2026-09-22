"""Summarize GPU provider benchmark JSON files into compact comparisons."""

import argparse
import json
from pathlib import Path

PROVIDERS = ("cuda", "tensorrt", "webgpu", "intel", "directml")
COMPARABLE = (
    "repo_id", "metadata_version", "seed", "image_size", "warmup",
    "iterations_per_set", "sets", "input_shape", "output_shape",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--providers", nargs="+", choices=PROVIDERS, default=["cuda", "webgpu"])
    parser.add_argument("--reference", choices=PROVIDERS, default="cuda")
    args = parser.parse_args()
    if args.reference not in args.providers:
        parser.error("--reference must be included in --providers")
    rows = {}
    for path in sorted(args.input_dir.glob("*.json")):
        if path.resolve() == args.output.resolve():
            continue
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if "profile" in row and "provider_requested" in row:
            key = (row["profile"], row["batch_size"], row["provider_requested"])
            if key in rows:
                parser.error(f"duplicate condition: {key}")
            rows[key] = row
    conditions = sorted({(profile, batch) for profile, batch, _ in rows})
    summary = {"reference": args.reference, "providers": args.providers,
               "results": [], "comparisons": [], "incomplete": []}
    for profile, batch in conditions:
        reference = rows.get((profile, batch, args.reference))
        for provider in args.providers:
            row = rows.get((profile, batch, provider))
            if row is None or row.get("status") != "ok":
                summary["incomplete"].append({
                    "profile": profile, "batch_size": batch, "provider": provider,
                    "reason": row.get("error", row.get("status")) if row else "missing",
                })
                continue
            summary["results"].append({
                "profile": profile, "batch_size": batch, "provider": provider,
                "ms_per_image": round(row["median_ms_per_image"], 3),
                "vram_increment_mib": row.get("vram_increment_mib"),
                "peak_rss_mib": row.get("peak_rss_mib"),
                "active": row["provider_active"],
                "executed_node_providers": row.get("executed_node_providers"),
                "onnxruntime": row["onnxruntime"],
            })
            if provider == args.reference:
                continue
            if reference is None or reference.get("status") != "ok":
                continue
            mismatched = [name for name in COMPARABLE
                          if row.get(name) != reference.get(name)]
            if (args.reference == "cuda" and provider in ("webgpu", "directml")
                    and not row.get("nvidia_gpu_memory_observed")):
                summary["incomplete"].append({
                    "profile": profile, "batch_size": batch, "provider": provider,
                    "reason": f"{provider} NVIDIA VRAM increment not observed",
                })
                continue
            if mismatched:
                summary["incomplete"].append({
                    "profile": profile, "batch_size": batch, "provider": provider,
                    "reason": f"mismatched inputs: {', '.join(mismatched)}",
                })
                continue
            reference_ms = reference["median_ms_per_image"]
            provider_ms = row["median_ms_per_image"]
            comparison = {
                "profile": profile, "batch_size": batch,
                "reference": args.reference, "provider": provider,
                "reference_ms_per_image": round(reference_ms, 3),
                "provider_ms_per_image": round(provider_ms, 3),
                "ratio": round(provider_ms / reference_ms, 3),
                "slowdown_pct": round((provider_ms / reference_ms - 1) * 100, 1),
            }
            if args.reference == "cuda" and provider == "webgpu":
                comparison.update({
                    "cuda_ms_per_image": round(reference_ms, 3),
                    "webgpu_ms_per_image": round(provider_ms, 3),
                    "webgpu_over_cuda_ratio": comparison["ratio"],
                    "webgpu_slowdown_pct": comparison["slowdown_pct"],
                    "cuda_vram_increment_mib": reference.get("vram_increment_mib"),
                    "webgpu_vram_increment_mib": row.get("vram_increment_mib"),
                    "webgpu_nvidia_memory_observed": row.get("nvidia_gpu_memory_observed"),
                    "cuda_active": reference["provider_active"],
                    "webgpu_active": row["provider_active"],
                    "cuda_version": reference["onnxruntime"],
                    "webgpu_version": row["onnxruntime"],
                })
            summary["comparisons"].append(comparison)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(f"Saved {args.output}: {len(summary['comparisons'])} comparisons, "
          f"{len(summary['incomplete'])} incomplete")


if __name__ == "__main__":
    main()
