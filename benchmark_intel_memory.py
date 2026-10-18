"""Compare OpenVINO ultra memory in fresh processes (local ONNX file).

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
    if os.name == "nt":
        return windows_snapshot(pid)
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


def windows_snapshot(pid):
    """Windows working set/private commit; commit is not physical swap usage."""
    import ctypes
    from ctypes import wintypes

    class MemoryStatus(ctypes.Structure):
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_ulonglong) for name in (
                "total_physical", "available_physical", "commit_limit", "available_commit",
                "total_virtual", "available_virtual", "available_extended")]

    class ProcessMemory(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in (
                "peak_working_set", "working_set", "peak_paged_pool", "paged_pool",
                "peak_nonpaged_pool", "nonpaged_pool", "pagefile", "peak_pagefile", "private")]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    psapi = ctypes.WinDLL("psapi", use_last_error=True)
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(MemoryStatus)]
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemory), wintypes.DWORD]
    status = MemoryStatus()
    status.length = ctypes.sizeof(status)
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ctypes.WinError(ctypes.get_last_error())
    mib = 1024 * 1024
    result = {"available_mib": status.available_physical / mib,
              "system_commit_mib": (status.commit_limit - status.available_commit) / mib,
              "commit_limit_mib": status.commit_limit / mib}
    # Windows venv python.exe may be a launcher for another interpreter process.
    for process_id in windows_process_ids(pid):
        handle = kernel.OpenProcess(0x0400 | 0x0010, False, process_id)
        if handle:
            try:
                counters = ProcessMemory()
                counters.cb = ctypes.sizeof(counters)
                if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
                    result["rss_mib"] = result.get("rss_mib", 0) + counters.working_set / mib
                    result["private_commit_mib"] = result.get("private_commit_mib", 0) + counters.private / mib
            finally:
                kernel.CloseHandle(handle)
    return result


def windows_process_ids(pid):
    import ctypes
    from ctypes import wintypes

    class Entry(ctypes.Structure):
        _fields_ = [("size", wintypes.DWORD), ("usage", wintypes.DWORD),
                    ("pid", wintypes.DWORD), ("heap", ctypes.c_size_t),
                    ("module", wintypes.DWORD), ("threads", wintypes.DWORD),
                    ("parent", wintypes.DWORD), ("priority", wintypes.LONG),
                    ("flags", wintypes.DWORD), ("name", wintypes.WCHAR * 260)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    kernel.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel.Process32FirstW.argtypes = kernel.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(Entry)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateToolhelp32Snapshot(2, 0)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    parents = {}
    try:
        entry = Entry()
        entry.size = ctypes.sizeof(entry)
        found = kernel.Process32FirstW(handle, ctypes.byref(entry))
        while found:
            parents[entry.pid] = entry.parent
            found = kernel.Process32NextW(handle, ctypes.byref(entry))
    finally:
        kernel.CloseHandle(handle)
    tracked = {pid}
    while True:
        expanded = tracked | {child for child, parent in parents.items() if parent in tracked}
        if expanded == tracked:
            return sorted(tracked)
        tracked = expanded


def child(args):
    import onnxruntime as ort
    import gpu_runtime

    device_name = gpu_runtime.prepare_openvino(args.device)
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
    if "OpenVINOExecutionProvider" not in session.get_providers():
        raise RuntimeError(f"OpenVINO EP did not initialize: {session.get_providers()}")
    load_seconds = time.monotonic() - started
    Path(args.child, "loaded").touch()
    Path(args.child, "progress.json").write_text(json.dumps({"stage": "loaded", "load_seconds": load_seconds}))
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
        Path(args.child, "progress.json").write_text(json.dumps({
            "stage": "inference", "completed_iterations": index + 1,
            "load_seconds": load_seconds, "inference_seconds": durations}))
        if index == 0:
            profile = session.end_profiling()
            executed = gpu_runtime.executed_providers(profile)
            if "OpenVINOExecutionProvider" not in executed:
                raise RuntimeError(f"GPU executed no nodes: {executed}")
    np.save(Path(args.child, "output.npy"), y)
    Path(args.child, "result.json").write_text(json.dumps({
        "load_seconds": load_seconds, "inference_seconds": durations,
        "executed_providers": sorted(executed), "output_shape": list(y.shape),
        "ort_version": ort.__version__, "batch_size": batch_size, "device_name": device_name,
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
        ], stdout=log, stderr=subprocess.STDOUT, start_new_session=os.name != "nt")
        timed_out = False
        try:
            while process.poll() is None:
                samples.append(snapshot(process.pid) | {"elapsed_seconds": time.monotonic() - started})
                if loaded_sample is None and (directory / "loaded").exists():
                    loaded_sample = samples[-1]
                if time.monotonic() - started > args.timeout:
                    timed_out = True
                    break
                time.sleep(0.05)
        finally:
            if process.poll() is None:
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                else:
                    import signal
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # The child completed between poll() and termination.
            process.wait()
    if process.returncode:
        result = {"status": "timeout" if timed_out else "failed", "exit_code": process.returncode}
        if (directory / "progress.json").exists():
            result["progress"] = json.loads((directory / "progress.json").read_text())
    else:
        result = json.loads((directory / "result.json").read_text()) | {"status": "success"}
    result.update(mode=mode, baseline=baseline, loaded=loaded_sample,
                  peak_rss_mib=max(s.get("rss_mib", 0) for s in samples),
                  min_available_mib=min(s["available_mib"] for s in samples))
    if "swap_free_mib" in baseline:
        result.update(peak_process_swap_mib=max(s.get("process_swap_mib", 0) for s in samples),
                      max_swap_growth_mib=baseline["swap_free_mib"] - min(s["swap_free_mib"] for s in samples))
    else:
        result.update(peak_private_commit_mib=max(s.get("private_commit_mib", 0) for s in samples),
                      max_system_commit_growth_mib=max(s["system_commit_mib"] for s in samples) - baseline["system_commit_mib"])
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
            if all(run["status"] == "success" for run in runs):
                before, after = [np.load(Path(temp) / mode / "output.npy") for mode in ("baseline", "current")]
                comparison = {"status": "compared", "max_abs_difference": float(np.max(np.abs(before - after))),
                              "allclose_rtol_1e-4_atol_1e-5": bool(np.allclose(before, after, rtol=1e-4, atol=1e-5))}
            else:
                comparison = {"status": "not_compared", "reason": "At least one run failed or timed out"}
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
    if not comparison.get("allclose_rtol_1e-4_atol_1e-5", False):
        raise SystemExit("Output comparison failed or was not possible; see summary.json")


if __name__ == "__main__":
    main()
