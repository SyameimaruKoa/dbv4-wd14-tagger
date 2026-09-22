"""Run one repeatable GPU provider benchmark and write its raw JSON result.

Run each provider in its own virtual environment; see WEBGPU_BENCHMARK_WINDOWS.md
and WEBGPU_BENCHMARK_LINUX.md. This script does not change user images.
"""

import argparse
import ctypes
import json
import os
import platform
import statistics
import subprocess
import sys
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



def prepare_tensorrt_windows(explicit_dir):
    """Load TensorRT and its ORT bridge before session creation on Windows."""
    name = "nvinfer_10.dll"
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    elif os.environ.get("TENSORRT_LIB_DIR"):
        candidates.append(Path(os.environ["TENSORRT_LIB_DIR"]))
    else:
        candidates.extend(Path.home().glob("Downloads/TensorRT-*/bin"))
        candidates.extend(Path(part) for part in os.environ.get("PATH", "").split(os.pathsep)
                          if part)
    matches = []
    for candidate in candidates:
        directory = candidate.resolve()
        if (directory / name).is_file() and directory not in matches:
            matches.append(directory)
    if not matches:
        raise RuntimeError(
            "TensorRT 10 runtime missing: nvinfer_10.dll not found. "
            "Pass --tensorrt-lib-dir with the TensorRT bin directory or set TENSORRT_LIB_DIR."
        )
    if len(matches) > 1 and not explicit_dir and not os.environ.get("TENSORRT_LIB_DIR"):
        raise RuntimeError(
            f"Multiple TensorRT runtimes found: {matches}; choose --tensorrt-lib-dir"
        )
    directory = matches[0]
    directory_handle = os.add_dll_directory(str(directory))
    os.environ["PATH"] = str(directory) + os.pathsep + os.environ.get("PATH", "")
    try:
        runtime_handle = ctypes.WinDLL(str(directory / name))
        provider_dll = Path(ort.__file__).resolve().parent / "capi" / "onnxruntime_providers_tensorrt.dll"
        provider_handle = ctypes.WinDLL(str(provider_dll))
    except OSError as exc:
        directory_handle.close()
        raise RuntimeError(f"TensorRT DLL preflight failed in {directory}: {exc}") from exc
    return directory, (directory_handle, runtime_handle, provider_handle)


def find_tensorrt_linux_dir(explicit_dir):
    """Find TensorRT 10 in an explicit, pip-venv, or system library directory."""
    name = "libnvinfer.so.10"
    if explicit_dir:
        candidates = [Path(explicit_dir)]
    elif os.environ.get("TENSORRT_LIB_DIR"):
        candidates = [Path(os.environ["TENSORRT_LIB_DIR"])]
    else:
        candidates = []
        venv = Path(sys.prefix)
        for pattern in ("lib/python*/site-packages/tensorrt*_libs",
                        "lib/python*/site-packages/tensorrt*_libs/lib",
                        "lib/python*/site-packages/tensorrt_libs"):
            candidates.extend(venv.glob(pattern))
        candidates.extend(Path(part) for part in
                          os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part)
        candidates.extend((Path("/usr/lib/x86_64-linux-gnu"),
                           Path("/usr/lib/aarch64-linux-gnu"), Path("/usr/local/lib")))
    matches = []
    for candidate in candidates:
        directory = candidate.resolve()
        if (directory / name).is_file() and directory not in matches:
            matches.append(directory)
    if len(matches) > 1 and not explicit_dir and not os.environ.get("TENSORRT_LIB_DIR"):
        raise RuntimeError(
            f"Multiple TensorRT 10 runtimes found: {matches}; choose --tensorrt-lib-dir"
        )
    return matches[0] if matches else None


def prepare_tensorrt_linux(explicit_dir):
    directory = find_tensorrt_linux_dir(explicit_dir)
    if (explicit_dir or os.environ.get("TENSORRT_LIB_DIR")) and directory is None:
        raise RuntimeError(
            "TensorRT 10 runtime missing in selected directory: libnvinfer.so.10"
        )
    source = str(directory / "libnvinfer.so.10") if directory else "libnvinfer.so.10"
    try:
        runtime_handle = ctypes.CDLL(source, mode=ctypes.RTLD_GLOBAL)
        plugin_handle = None
        if directory and (directory / "libnvinfer_plugin.so.10").is_file():
            plugin_handle = ctypes.CDLL(
                str(directory / "libnvinfer_plugin.so.10"), mode=ctypes.RTLD_GLOBAL
            )
        provider_so = Path(ort.__file__).resolve().parent / "capi" / "libonnxruntime_providers_tensorrt.so"
        provider_handle = ctypes.CDLL(str(provider_so), mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        raise RuntimeError(
            f"TensorRT Linux preflight failed ({source}). Set --tensorrt-lib-dir "
            f"or install matching tensorrt-cu12 libraries: {exc}"
        ) from exc
    return directory, (runtime_handle, plugin_handle, provider_handle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("cpu", "cuda", "tensorrt", "webgpu", "intel", "directml", "migraphx"), required=True)
    parser.add_argument("--openvino-device", default="GPU.0")
    parser.add_argument("--tensorrt-lib-dir", type=Path,
                        help="TensorRT 10 bin directory; auto-detected on Windows when omitted")
    parser.add_argument("--target-vendor", choices=("cpu", "intel", "nvidia", "amd"))
    parser.add_argument("--device-name", help="Operator-confirmed adapter name for DirectML/WebGPU/MIGraphX")
    parser.add_argument("--profile", choices=tuple(MODEL_PROFILES), required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--directml-device-index", type=int, default=0)
    parser.add_argument("--track-nvidia-memory", action="store_true")
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

    track_nvidia = args.provider in ("cuda", "tensorrt") or args.track_nvidia_memory
    if args.provider == "cpu" and args.target_vendor not in (None, "cpu"):
        parser.error("CPU provider requires target vendor cpu")
    if args.provider == "intel" and args.target_vendor not in (None, "intel"):
        parser.error("OpenVINO Intel provider requires target vendor intel")
    if args.provider in ("directml", "webgpu", "migraphx") and args.target_vendor and not args.device_name:
        parser.error("this provider requires --device-name when --target-vendor is set")
    if args.provider in ("directml", "webgpu", "migraphx") and args.target_vendor:
        if args.target_vendor not in args.device_name.lower():
            parser.error("--device-name must include the target vendor")
    if args.provider in ("cuda", "tensorrt") and args.target_vendor not in (None, "nvidia"):
        parser.error("CUDA/TensorRT require target vendor nvidia")
    try:
        baseline_vram = nvidia_memory_mib(args.gpu_index) if track_nvidia else None
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        parser.error(f"nvidia-smi cannot read GPU {args.gpu_index}: {exc}")
    baseline_rss = psutil.Process().memory_info().rss / 1048576
    samples_vram = [baseline_vram] if track_nvidia else []
    samples_rss = [baseline_rss]
    stop = threading.Event()

    def monitor():
        process = psutil.Process()
        while not stop.wait(0.1):
            try:
                if track_nvidia:
                    samples_vram.append(nvidia_memory_mib(args.gpu_index))
                samples_rss.append(process.memory_info().rss / 1048576)
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    watcher = threading.Thread(target=monitor, daemon=True)
    watcher.start()
    session = None
    profile_stopped = False
    webgpu_hardware = None
    tensorrt_lib_dir = None
    tensorrt_handles = None
    try:
        if args.provider == "tensorrt":
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
            if platform.system() == "Windows":
                tensorrt_lib_dir, tensorrt_handles = prepare_tensorrt_windows(
                    args.tensorrt_lib_dir
                )
            elif platform.system() == "Linux":
                tensorrt_lib_dir, tensorrt_handles = prepare_tensorrt_linux(
                    args.tensorrt_lib_dir
                )
        started = time.perf_counter()
        metadata = DBV4Metadata.load(profile, base_dir=str(Path(__file__).resolve().parent))
        metadata_seconds = time.perf_counter() - started
        dll_directory_handle = None
        tensor_ir_handle = None
        if args.provider in ("cuda", "tensorrt") and hasattr(ort, "preload_dlls"):
            ort.preload_dlls()
            if platform.system() == "Windows":
                cudnn_bin = (Path(ort.__file__).resolve().parent.parent
                             / "nvidia" / "cudnn" / "bin")
                if cudnn_bin.is_dir():
                    dll_directory_handle = os.add_dll_directory(str(cudnn_bin))
                    os.environ["PATH"] = str(cudnn_bin) + os.pathsep + os.environ.get("PATH", "")
                    tensor_ir = cudnn_bin / "cudnn_engines_tensor_ir64_9.dll"
                    if tensor_ir.is_file():
                        tensor_ir_handle = ctypes.WinDLL(str(tensor_ir))
        if args.provider == "intel" and platform.system() == "Windows":
            openvino_libs = (Path(ort.__file__).resolve().parent.parent
                             / "openvino" / "libs")
            openvino_dll = openvino_libs / "openvino.dll"
            if openvino_dll.is_file():
                dll_directory_handle = os.add_dll_directory(str(openvino_libs))
                os.environ["PATH"] = str(openvino_libs) + os.pathsep + os.environ.get("PATH", "")
                openvino_handle = ctypes.WinDLL(str(openvino_dll))
            else:
                try:
                    openvino_handle = ctypes.WinDLL("openvino.dll")
                except OSError as exc:
                    raise RuntimeError(
                        "OpenVINO runtime missing; install openvino==2025.4.1 "
                        "in the Intel benchmark environment"
                    ) from exc
        openvino_gpu_name = None
        if args.provider == "intel":
            if not args.openvino_device.upper().startswith("GPU"):
                raise RuntimeError("Intel benchmark requires an OpenVINO GPU device")
            import openvino as ov

            openvino_gpu_name = ov.Core().get_property(
                args.openvino_device, "FULL_DEVICE_NAME"
            )
            if "intel" not in openvino_gpu_name.lower():
                raise RuntimeError(
                    f"OpenVINO {args.openvino_device} is {openvino_gpu_name}; "
                    "an Intel GPU is required for the Intel benchmark"
                )
        options = ort.SessionOptions()
        options.enable_profiling = True
        if args.provider in ("cpu", "cuda", "tensorrt", "intel", "directml", "migraphx"):
            if args.provider == "directml":
                options.enable_mem_pattern = False
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            available = ort.get_available_providers()
            names = {
                "cuda": "CUDAExecutionProvider",
                "tensorrt": "TensorrtExecutionProvider",
                "intel": "OpenVINOExecutionProvider",
                "directml": "DmlExecutionProvider",
                "migraphx": "MIGraphXExecutionProvider",
                "cpu": "CPUExecutionProvider",
            }
            expected_provider = names[args.provider]
            if expected_provider not in available:
                raise RuntimeError(f"{expected_provider} unavailable: {available}")
            if args.provider == "cpu":
                providers = [expected_provider]
            elif args.provider == "cuda":
                providers = [(expected_provider, {"device_id": str(args.gpu_index)}),
                             "CPUExecutionProvider"]
            elif args.provider == "tensorrt":
                providers = [(expected_provider, {"device_id": str(args.gpu_index)}),
                             ("CUDAExecutionProvider", {"device_id": str(args.gpu_index)}),
                             "CPUExecutionProvider"]
            elif args.provider == "intel":
                providers = [(expected_provider, {"device_type": args.openvino_device}),
                             "CPUExecutionProvider"]
            elif args.provider == "migraphx":
                providers = [(expected_provider, {"device_id": str(args.gpu_index)}),
                             "CPUExecutionProvider"]
            else:
                providers = [(expected_provider, {"device_id": str(args.directml_device_index)}),
                             "CPUExecutionProvider"]
            started = time.perf_counter()
            session = ort.InferenceSession(metadata.model_path, sess_options=options,
                                           providers=providers)
        else:
            import onnxruntime_ep_webgpu as webgpu

            ort.register_execution_provider_library("benchmark_webgpu", webgpu.get_library_path())
            devices = [device for device in ort.get_ep_devices()
                       if device.ep_name == webgpu.get_ep_name()]
            if args.webgpu_device_index < 0 or args.webgpu_device_index >= len(devices):
                raise RuntimeError(f"WebGPU device index unavailable: {len(devices)} devices")
            selected = devices[args.webgpu_device_index].device
            webgpu_hardware = {
                "vendor": selected.vendor,
                "vendor_id": selected.vendor_id,
                "device_id": selected.device_id,
                "metadata": dict(selected.metadata),
            }
            if args.target_vendor and args.target_vendor not in selected.vendor.lower():
                raise RuntimeError(
                    f"WebGPU device {args.webgpu_device_index} is "
                    f"{selected.vendor} {selected.metadata.get('Description')}; "
                    f"expected {args.target_vendor}"
                )
            options.add_provider_for_devices([devices[args.webgpu_device_index]], {})
            started = time.perf_counter()
            session = ort.InferenceSession(metadata.model_path, sess_options=options)
            expected_provider = webgpu.get_ep_name()
        session_seconds = time.perf_counter() - started
        session.disable_fallback()
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
        peak_vram = max(samples_vram + [nvidia_memory_mib(args.gpu_index)]) if track_nvidia else None
        profile_path = Path(session.end_profiling())
        profile_stopped = True
        try:
            events = json.loads(profile_path.read_text(encoding="utf-8"))
            executed = sorted({
                event.get("args", {}).get("provider")
                for event in events
                if event.get("cat") == "Node"
                and event.get("args", {}).get("provider")
            })
        finally:
            profile_path.unlink(missing_ok=True)
        if expected_provider not in executed:
            raise RuntimeError(f"{expected_provider} did not execute a profiled node: {executed}")
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
            "vram_increment_mib": peak_vram - baseline_vram if track_nvidia else None,
            "baseline_rss_mib": baseline_rss, "peak_rss_mib": max(samples_rss),
            "metadata_seconds": metadata_seconds, "session_seconds": session_seconds,
            "platform": platform.platform(), "python": platform.python_version(),
            "onnxruntime": ort.__version__,
            "webgpu_plugin": version("onnxruntime-ep-webgpu") if args.provider == "webgpu" else None,
            "gpu_index": args.gpu_index,
            "directml_device_index": args.directml_device_index if args.provider == "directml" else None,
            "webgpu_device_index": args.webgpu_device_index if args.provider == "webgpu" else None,
            "nvidia_gpu_memory_observed": peak_vram - baseline_vram >= 16 if track_nvidia else None,
            "executed_node_providers": executed,
            "openvino_device": args.openvino_device if args.provider == "intel" else None,
            "openvino_gpu_name": openvino_gpu_name,
            "webgpu_hardware": webgpu_hardware,
            "tensorrt_lib_dir": str(tensorrt_lib_dir) if tensorrt_lib_dir else None,
            "target_vendor": args.target_vendor,
            "device_name": openvino_gpu_name if args.provider == "intel" else args.device_name,
        }
    except Exception as exc:
        record = {
            "status": "failed", "provider_requested": args.provider,
            "profile": args.profile, "batch_size": args.batch_size,
            "target_vendor": args.target_vendor, "device_name": args.device_name,
            "error": f"{type(exc).__name__}: {exc}",
            "platform": platform.platform(), "onnxruntime": ort.__version__,
        }
    finally:
        stop.set()
        watcher.join(timeout=2)
        if session is not None and not profile_stopped:
            try:
                Path(session.end_profiling()).unlink(missing_ok=True)
            except (OSError, RuntimeError):
                pass
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    if record["status"] == "ok":
        print(f"Saved {args.output}: {record['median_ms_per_image']:.2f} ms/image")
    else:
        raise RuntimeError(f"Saved failure details to {args.output}: {record['error']}")


if __name__ == "__main__":
    main()
