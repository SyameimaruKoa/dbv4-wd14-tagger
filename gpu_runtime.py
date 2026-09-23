"""GPU initialization shared by normal inference and benchmark runners."""

import ctypes
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import onnxruntime as ort
from webgpu_vendor import vendor_name

# Keep DLL search paths and loaded libraries alive for all live sessions.
_RUNTIME_HANDLES = []

def prepare_tensorrt_windows(explicit_dir):
    """Load TensorRT and its ORT bridge before session creation on Windows."""
    name = "nvinfer_10.dll"
    prepare_cuda()
    candidates = []
    if explicit_dir:
        candidates.append(Path(explicit_dir))
    elif os.environ.get("TENSORRT_LIB_DIR"):
        candidates.append(Path(os.environ["TENSORRT_LIB_DIR"]))
    else:
        venv = Path(sys.prefix)
        candidates.extend(venv.glob("Lib/site-packages/tensorrt*_libs"))
        candidates.extend(venv.glob("Lib/site-packages/tensorrt*_libs/lib"))
        portable_runtime = Path(__file__).resolve().parent / ".dbv4" / "runtime"
        candidates.extend(portable_runtime.glob("TensorRT-*/bin"))
        candidates.append(portable_runtime / "tensorrt" / "bin")
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


def prepare_cuda():
    """Preload CUDA/cuDNN, including the Windows tensor IR dependency."""
    handles = []
    if hasattr(ort, "preload_dlls"):
        ort.preload_dlls()
    if platform.system() == "Windows":
        cudnn_bin = Path(ort.__file__).resolve().parent.parent / "nvidia" / "cudnn" / "bin"
        if cudnn_bin.is_dir():
            handles.append(os.add_dll_directory(str(cudnn_bin)))
            os.environ["PATH"] = str(cudnn_bin) + os.pathsep + os.environ.get("PATH", "")
            tensor_ir = cudnn_bin / "cudnn_engines_tensor_ir64_9.dll"
            if tensor_ir.is_file():
                handles.append(ctypes.WinDLL(str(tensor_ir)))
    _RUNTIME_HANDLES.extend(handles)
    return handles


def prepare_tensorrt(explicit_dir=None):
    prepare_cuda()
    prepare = prepare_tensorrt_windows if platform.system() == "Windows" else prepare_tensorrt_linux
    directory, handles = prepare(explicit_dir)
    _RUNTIME_HANDLES.extend(handles)
    return directory


def tensorrt_provider_options(gpu_index):
    """Return TensorRT options with a repository-local engine cache."""
    cache_dir = Path(os.environ.get(
        "TENSORRT_ENGINE_CACHE_DIR",
        Path(__file__).resolve().parent / ".dbv4" / "tensorrt-engine-cache",
    )).resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "device_id": str(gpu_index),
        "trt_engine_cache_enable": "1",
        "trt_engine_cache_path": str(cache_dir),
    }


def prepare_openvino(device):
    if not device.upper().startswith("GPU"):
        raise RuntimeError("Intel provider requires an OpenVINO GPU device")
    if platform.system() == "Windows":
        libs = Path(ort.__file__).resolve().parent.parent / "openvino" / "libs"
        if (libs / "openvino.dll").is_file():
            _RUNTIME_HANDLES.append(os.add_dll_directory(str(libs)))
            os.environ["PATH"] = str(libs) + os.pathsep + os.environ.get("PATH", "")
            source = str(libs / "openvino.dll")
        else:
            source = "openvino.dll"
        try:
            _RUNTIME_HANDLES.append(ctypes.WinDLL(source))
        except OSError as exc:
            raise RuntimeError("OpenVINO runtime missing; install openvino==2025.4.1") from exc
    # Keep the OpenVINO Python runtime outside the ORT process (runtime conflict).
    query = "import openvino as ov, sys; print(ov.Core().get_property(sys.argv[1], 'FULL_DEVICE_NAME'))"
    try:
        name = subprocess.check_output([sys.executable, "-c", query, device], text=True, timeout=30).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Cannot inspect OpenVINO {device}: {exc}") from exc
    if "intel" not in name.lower():
        raise RuntimeError(f"OpenVINO {device} is {name}; an Intel GPU is required")
    return name


def configure_directml(options):
    options.enable_mem_pattern = False
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    # ORT 1.24's optimizer can detach CAFormer's logits output.
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL


def webgpu_devices():
    import onnxruntime_ep_webgpu as webgpu
    # Multiple models can be loaded in a server process.
    if not getattr(webgpu_devices, "registered", False):
        ort.register_execution_provider_library("dbv4_webgpu", webgpu.get_library_path())
        webgpu_devices.registered = True
    return [d for d in ort.get_ep_devices() if d.ep_name == webgpu.get_ep_name()]


def configure_webgpu(options, index=None, target_vendor=None):
    devices = webgpu_devices()
    matches = [i for i, d in enumerate(devices)
               if not target_vendor or vendor_name(d.device.vendor, d.device.vendor_id) == target_vendor]
    if index is None:
        if len(matches) != 1:
            raise RuntimeError(f"WebGPU selection is ambiguous or unavailable ({matches}); specify --webgpu-device-index")
        index = matches[0]
    if index < 0 or index >= len(devices):
        raise RuntimeError(f"WebGPU device index unavailable: {index}; {len(devices)} devices")
    d = devices[index].device
    vendor = vendor_name(d.vendor, d.vendor_id)
    if target_vendor and vendor != target_vendor:
        raise RuntimeError(f"WebGPU device {index} is {vendor}; expected {target_vendor}")
    options.add_provider_for_devices([devices[index]], {})
    return {"vendor": vendor, "vendor_id": d.vendor_id, "device_id": d.device_id,
            "metadata": dict(d.metadata)}


def validate_directml(index, target_vendor):
    """Check DXGI identity in the separate WebGPU environment (ORT versions differ)."""
    if not target_vendor:
        return
    root = Path(__file__).resolve().parent
    executable = root / "venv_webgpu" / "Scripts" / "python.exe"
    if not executable.is_file():
        raise RuntimeError("DirectML vendor validation requires venv_webgpu; use run_tagger.ps1 -Provider directml -TargetVendor")
    try:
        records = json.loads(subprocess.check_output(
            [str(executable), str(root / "probe_webgpu_adapters.py"), "--json"],
            text=True, timeout=30))
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(f"Cannot inspect DirectML DXGI adapters: {exc}") from exc
    matches = [r for r in records if r["metadata"].get("DxgiAdapterNumber") == str(index)]
    if not matches or any(vendor_name(r["vendor"], r["vendor_id"]) != target_vendor for r in matches):
        raise RuntimeError(f"DirectML DXGI {index} cannot be confirmed as {target_vendor}")


def executed_providers(path):
    events = json.loads(Path(path).read_text(encoding="utf-8"))
    return sorted({e.get("args", {}).get("provider") for e in events
                   if e.get("cat") == "Node" and e.get("args", {}).get("provider")})
