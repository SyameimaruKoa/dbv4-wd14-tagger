"""Summarize a platform/vendor benchmark matrix, keeping failures visible."""

import argparse
import json
from pathlib import Path

COMPARABLE = (
    "repo_id", "metadata_version", "seed", "image_size", "warmup",
    "iterations_per_set", "sets", "input_shape", "output_shape",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text(encoding="utf-8"))
    summary = {key: manifest[key] for key in (
        "platform", "vendor", "operator_device_name", "providers", "profiles",
        "batches", "warmup", "iterations", "sets"
    )}
    summary.update({"results": [], "comparisons_to_cpu": [], "incomplete": []})
    rows = {}
    for source in manifest["results"]:
        path = Path(source)
        row = json.loads(path.read_text(encoding="utf-8"))
        key = (row["profile"], row["batch_size"], row["provider_requested"])
        rows[key] = row
        if row["status"] != "ok":
            summary["incomplete"].append({
                "profile": key[0], "batch_size": key[1], "provider": key[2],
                "reason": row.get("error", row["status"]),
            })
            continue
        summary["results"].append({
            "profile": key[0], "batch_size": key[1], "provider": key[2],
            "ms_per_image": row["median_ms_per_image"],
            "peak_rss_mib": row.get("peak_rss_mib"),
            "vram_increment_mib": row.get("vram_increment_mib"),
            "device_name": row.get("device_name"),
            "executed_node_providers": row.get("executed_node_providers"),
        })
    for profile in manifest["profiles"]:
        for batch in manifest["batches"]:
            cpu = rows.get((profile, batch, "cpu"))
            for provider in manifest["providers"]:
                if provider == "cpu":
                    continue
                row = rows.get((profile, batch, provider))
                if not row or row["status"] != "ok":
                    continue
                if not cpu or cpu["status"] != "ok":
                    continue
                mismatch = [field for field in COMPARABLE if row.get(field) != cpu.get(field)]
                if mismatch:
                    summary["incomplete"].append({
                        "profile": profile, "batch_size": batch, "provider": provider,
                        "reason": f"CPU comparison has mismatched {', '.join(mismatch)}",
                    })
                    continue
                summary["comparisons_to_cpu"].append({
                    "profile": profile, "batch_size": batch, "provider": provider,
                    "gpu_ms_per_image": row["median_ms_per_image"],
                    "cpu_ms_per_image": cpu["median_ms_per_image"],
                    "speedup_over_cpu": round(
                        cpu["median_ms_per_image"] / row["median_ms_per_image"], 3
                    ),
                })
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {args.output}: {len(summary['results'])} results, "
          f"{len(summary['incomplete'])} incomplete")


if __name__ == "__main__":
    main()
