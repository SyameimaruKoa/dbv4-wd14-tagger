"""Compare OpenVINO ultra memory in fresh processes (Linux, local ONNX file).

RSS and process swap exclude some driver allocations, so system available RAM
and swap are sampled as well. Other running applications affect system counters.
The synthetic input tests numerical regression, not tagging quality.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

import numpy as np


def snapshot(pid):
    result = {}
    for path, keys in (
        (Path(f"/proc/{pid}/status"), {"VmRSS": "rss_mib", "VmSwap": "process_swap_mib"}),
        (Path("/proc/meminfo"), {"MemAvailable": "available_mib", "SwapFree": "swap_free_mib"}),
    ):
        try:
            for line in path.read_text().splitlines():
                name, _, value = line.partition(":")
                if name in keys:
                    result[keys[name]] = int(value.split()[0]) / 1024
        except FileNotFoundError:
            pass
    return result


def child(args):
    import onnxruntime as ort
    import gpu_runtime

    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.enable_profiling = True
    options.profile_file_prefix = str(Path(args.child) / "ort")
    config = {"INFERENCE_PRECISION_HINT": "f32"}
    if args.mode == "current":
        gpu_runtime.configure_openvino(options)
        config.update(PERFORMANCE_HINT="LATENCY", NUM_STREAMS="1")
    started = time.monotonic()
    session = ort.InferenceSession(args.model, sess_options=options, providers=[
        ("OpenVINOExecutionProvider", {"device_type": args.device,
                                      "load_config": json.dumps({"GPU": config})}),
        "CPUExecutionProvider",
    ])
    session.disable_fallback()
    load_seconds = time.monotonic() - started
    Path(args.child, "loaded").touch()
    # DBV4 ultra uses NCHW RGB, 512 square, normalized around zero.
    x = np.random.default_rng(29).uniform(-1, 1, (1, 3, 512, 512)).astype(np.float32)
    batch_size = args.baseline_batch_size if args.mode == "baseline" else 1
    x = np.repeat(x, batch_size, axis=0)
    durations = []
    for index in range(args.iterations):
        started = time.monotonic()
        y = session.run(None, {session.get_inputs()[0].name: x})[0]
        durations.append(time.monotonic() - started)
        if not np.isfinite(y).all():
            raise RuntimeError("Non-finite output")
        if index == 0:
            profile = session.end_profiling()
            executed = gpu_runtime.executed_providers(profile)
            if "OpenVINOExecutionProvider" not in executed:
                raise RuntimeError(f"GPU executed no nodes: {executed}")
    np.save(Path(args.child, "output.npy"), y)
    Path(args.child, "result.json").write_text(json.dumps({
        "load_seconds": load_seconds, "inference_seconds": durations,
        "executed_providers": sorted(executed), "output_shape": list(y.shape),
        "ort_version": ort.__version__, "batch_size": batch_size,
    }))


def measure(args, mode, directory):
    directory.mkdir()
    baseline = snapshot(os.getpid())
    samples = []
    loaded_sample = None
    started = time.monotonic()
    with (directory / "run.log").open("w") as log:
        process = subprocess.Popen([
            sys.executable, __file__, args.model, "--device", args.device,
            "--iterations", str(args.iterations), "--mode", mode,
            "--baseline-batch-size", str(args.baseline_batch_size),
            "--child", str(directory),
        ], stdout=log, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                samples.append(snapshot(process.pid))
                if loaded_sample is None and (directory / "loaded").exists():
                    loaded_sample = samples[-1]
                if time.monotonic() - started > args.timeout:
                    raise TimeoutError(f"{mode} exceeded {args.timeout}s; see {directory / 'run.log'}")
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
    if process.returncode:
        raise RuntimeError(f"{mode} failed ({process.returncode}); see {directory / 'run.log'}")
    result = json.loads((directory / "result.json").read_text())
    result.update(mode=mode, baseline=baseline, loaded=loaded_sample,
                  peak_rss_mib=max(s.get("rss_mib", 0) for s in samples),
                  peak_process_swap_mib=max(s.get("process_swap_mib", 0) for s in samples),
                  min_available_mib=min(s["available_mib"] for s in samples),
                  max_swap_growth_mib=baseline["swap_free_mib"] - min(s["swap_free_mib"] for s in samples))
    (directory / "samples.json").write_text(json.dumps(samples))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="Local ultra model.onnx")
    parser.add_argument("--device", default="GPU.0")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--baseline-batch-size", type=int, default=1,
                        help="Use 4 to reproduce previous CLI default; may cause heavy swap")
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", default="benchmarks/intel_memory")
    parser.add_argument("--mode", choices=("baseline", "current"), default="current")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations < 1 or args.timeout <= 0 or args.baseline_batch_size < 1:
        parser.error("iterations, baseline batch size and timeout must be positive")
    if args.child:
        child(args)
        return
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="runs-", dir=output) as temp:
        try:
            runs = [measure(args, mode, Path(temp) / mode) for mode in ("baseline", "current")]
            before, after = [np.load(Path(temp) / mode / "output.npy") for mode in ("baseline", "current")]
            comparison = {"max_abs_difference": float(np.max(np.abs(before - after))),
                          "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(before, after, rtol=1e-4, atol=1e-5))}
        finally:
            # Preserve diagnostics even when compilation, inference or timeout fails.
            import shutil
            for mode in ("baseline", "current"):
                if (Path(temp) / mode).exists():
                    shutil.copytree(Path(temp) / mode, output / mode, dirs_exist_ok=True)
    report = {"model": str(Path(args.model).resolve()), "device": args.device,
              "sample_interval_seconds": 0.05, "runs": runs, "comparison": comparison}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not comparison["allclose_rtol_1e-4_atol_1e-5"]:
        raise SystemExit("Output comparison failed")


if __name__ == "__main__":
    main()
