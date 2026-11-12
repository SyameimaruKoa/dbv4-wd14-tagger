"""Detect physical RAM and dedicated or shared GPU memory for automatic low-memory mode."""

import ctypes
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from uuid import UUID

GIB = 1024 ** 3
RAM_LIMIT = 16 * GIB
VRAM_LIMIT = 8 * GIB

GPU_VENDORS = {
    "intel": 0x8086,
    "nvidia": 0x10DE,
    "amd": 0x1002,
}


def total_ram_bytes():
    system = platform.system()
    try:
        if system == "Linux":
            data = Path("/proc/meminfo").read_text(encoding="utf-8")
            match = re.search(r"^MemTotal:\s*(\d+)\s+kB", data, re.MULTILINE)
            if not match:
                return None
            total = int(match.group(1)) * 1024
            for path in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
                try:
                    limit = int(Path(path).read_text().strip())
                    if 0 < limit < total:
                        total = limit
                except (OSError, ValueError):
                    pass
            return total
        if system == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (field, ctypes.c_ulonglong) for field in (
                        "total_phys", "avail_phys", "total_page", "avail_page",
                        "total_virtual", "avail_virtual", "avail_extended"
                    )
                ]
            info = MemoryStatus()
            info.length = ctypes.sizeof(info)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(info)):
                return int(info.total_phys)
            return None
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return pages * page_size
    except (OSError, ValueError, AttributeError):
        return None


def _linux_cards():
    devices = []
    for card in sorted(Path("/sys/class/drm").glob("card[0-9]*"),
                       key=lambda path: int(path.name[4:]) if path.name[4:].isdigit() else 9999):
        if not card.name[4:].isdigit():
            continue
        device = card / "device"
        try:
            vendor = int((device / "vendor").read_text().strip(), 0)
        except (OSError, ValueError):
            continue
        try:
            vram = int((device / "mem_info_vram_total").read_text().strip())
        except (OSError, ValueError):
            vram = 0 if vendor == GPU_VENDORS["intel"] else None
        try:
            shared = int((device / "mem_info_gtt_total").read_text().strip())
        except (OSError, ValueError):
            shared = None
        devices.append({"vendor": vendor, "dedicated": vram,
                        "shared": shared, "name": card.name})
    return devices


def _windows_dxgi_adapters():
    class GUID(ctypes.Structure):
        _fields_ = [("data1", ctypes.c_uint32), ("data2", ctypes.c_uint16),
                    ("data3", ctypes.c_uint16), ("data4", ctypes.c_ubyte * 8)]

    class LUID(ctypes.Structure):
        _fields_ = [("low", ctypes.c_uint32), ("high", ctypes.c_int32)]

    class AdapterDesc(ctypes.Structure):
        _fields_ = [("description", ctypes.c_wchar * 128), ("vendor", ctypes.c_uint),
                    ("device", ctypes.c_uint), ("subsystem", ctypes.c_uint),
                    ("revision", ctypes.c_uint), ("dedicated_vram", ctypes.c_size_t),
                    ("dedicated_system", ctypes.c_size_t), ("shared_system", ctypes.c_size_t),
                    ("luid", LUID), ("flags", ctypes.c_uint)]

    identifier = UUID("770aae78-f26f-4dba-a829-253c83d1b387")
    guid = GUID(identifier.time_low, identifier.time_mid, identifier.time_hi_version,
                (ctypes.c_ubyte * 8).from_buffer_copy(identifier.bytes[8:]))
    dll = ctypes.WinDLL("dxgi.dll")
    create = dll.CreateDXGIFactory1
    create.argtypes = (ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p))
    create.restype = ctypes.c_long
    factory = ctypes.c_void_p()
    if create(ctypes.byref(guid), ctypes.byref(factory)) != 0:
        return []
    call = ctypes.WINFUNCTYPE

    def method(obj, slot, return_type, *types):
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return call(return_type, ctypes.c_void_p, *types)(table[slot])

    adapters = []
    try:
        enum_adapter = method(factory, 12, ctypes.c_long, ctypes.c_uint,
                              ctypes.POINTER(ctypes.c_void_p))
        index = 0
        while True:
            adapter = ctypes.c_void_p()
            if enum_adapter(factory, index, ctypes.byref(adapter)) != 0:
                break
            try:
                desc = AdapterDesc()
                get_desc = method(adapter, 10, ctypes.c_long, ctypes.POINTER(AdapterDesc))
                if get_desc(adapter, ctypes.byref(desc)) == 0:
                    adapters.append({"vendor": int(desc.vendor),
                                     "dedicated": int(desc.dedicated_vram),
                                     "shared": int(desc.shared_system),
                                     "name": desc.description, "index": index,
                                     "software": bool(desc.flags & 2)})
            finally:
                method(adapter, 2, ctypes.c_ulong)(adapter)
            index += 1
    finally:
        method(factory, 2, ctypes.c_ulong)(factory)
    return adapters


def _openvino_gpu_memory(device):
    query = (
        "import openvino as ov,sys; c=ov.Core(); d=sys.argv[1]; "
        "print(str(c.get_property(d, 'DEVICE_TYPE'))); "
        "print(c.get_property(d, 'GPU_DEVICE_TOTAL_MEM_SIZE'))"
    )
    try:
        result = subprocess.run([sys.executable, "-c", query, device or "GPU.0"],
                                capture_output=True, text=True, timeout=15, check=True)
        kind, capacity = result.stdout.strip().splitlines()[-2:]
        capacity = int(capacity)
        if capacity <= 0:
            return None, None
        kind = kind.lower()
        if "igpu" in kind or "integrated" in kind:
            return None, capacity
        if "dgpu" in kind or "discrete" in kind:
            return capacity, None
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        pass
    return None, None


def _nvidia_vram_bytes(index):
    try:
        result = subprocess.run(
            ["nvidia-smi", "-i", str(index), "--query-gpu=memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5, check=True,
        )
        return int(result.stdout.strip().splitlines()[0]) * 1024 ** 2
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def _selected_adapter(adapters, provider, gpu_index, directml_device_index,
                      openvino_device, target_vendor):
    adapters = [adapter for adapter in adapters if not adapter.get("software")]
    if not adapters:
        return None
    if provider == "directml":
        return next((adapter for adapter in adapters
                     if adapter.get("index") == directml_device_index), None)
    vendor = {"intel": "intel", "cuda": "nvidia", "tensorrt": "nvidia",
              "rocm": "amd", "migraphx": "amd"}.get(provider, target_vendor)
    if vendor in GPU_VENDORS:
        adapters = [adapter for adapter in adapters if adapter["vendor"] == GPU_VENDORS[vendor]]
    if not adapters:
        return None
    if provider == "intel":
        match = re.fullmatch(r"GPU(?:\.(\d+))?", openvino_device or "GPU.0", re.IGNORECASE)
        index = int(match.group(1) or 0) if match else 0
    elif provider in ("cuda", "tensorrt", "rocm", "migraphx"):
        index = gpu_index
    elif len(adapters) == 1:
        index = 0
    else:
        return None
    return adapters[index] if 0 <= index < len(adapters) else None


def select_gpu_memory(dedicated, shared, integrated=False):
    if integrated:
        if shared is not None and shared > 0:
            return shared, "共有GPUメモリ"
        return None, None
    if dedicated is not None and dedicated > 0:
        return dedicated, "専用VRAM"
    if shared is not None and shared > 0:
        return shared, "共有GPUメモリ"
    return None, None


def _is_integrated_adapter(adapter):
    if adapter is None:
        return False
    vendor = adapter.get("vendor")
    name = adapter.get("name", "").lower()
    if vendor == GPU_VENDORS["intel"]:
        if re.search(r"\b(?:hd graphics|uhd graphics|iris(?: xe)? graphics)\b", name):
            return True
        dedicated = adapter.get("dedicated")
        shared = adapter.get("shared")
        if dedicated is not None and 0 <= dedicated <= 256 * 1024 ** 2 and shared:
            return True
    return False


def gpu_memory_capacity(provider=None, gpu_index=0, directml_device_index=0,
                        openvino_device=None, target_vendor=None):
    if provider == "cpu":
        return None, None
    if provider in ("cuda", "tensorrt"):
        detected = _nvidia_vram_bytes(gpu_index)
        if detected is not None and detected > 0:
            return detected, "専用VRAM"

    dedicated = shared = None
    integrated = False
    try:
        system = platform.system()
        if system == "Linux":
            adapters = _linux_cards()
        elif system == "Windows":
            adapters = _windows_dxgi_adapters()
        else:
            adapters = []
        selected = _selected_adapter(adapters, provider, gpu_index, directml_device_index,
                                     openvino_device, target_vendor)
        if selected is not None:
            dedicated = selected.get("dedicated")
            shared = selected.get("shared")
            integrated = _is_integrated_adapter(selected)
    except (OSError, ValueError, AttributeError):
        pass

    if provider == "intel":
        openvino_dedicated, openvino_shared = _openvino_gpu_memory(openvino_device)
        if openvino_shared is not None and openvino_shared > 0:
            return select_gpu_memory(dedicated, openvino_shared, integrated=True)
        if openvino_dedicated is not None and openvino_dedicated > 0:
            return select_gpu_memory(openvino_dedicated, shared)
    return select_gpu_memory(dedicated, shared, integrated=integrated)


def detect_low_memory(args):
    ram = total_ram_bytes()
    vram, memory_type = gpu_memory_capacity(
        provider=getattr(args, "provider", None),
        gpu_index=getattr(args, "gpu_index", 0),
        directml_device_index=getattr(args, "directml_device_index", 0),
        openvino_device=getattr(args, "openvino_device", None),
        target_vendor=getattr(args, "target_vendor", None),
    ) if getattr(args, "gpu", False) or getattr(args, "provider", None) not in (None, "cpu") else (None, None)
    reasons = []
    if ram is not None and ram < RAM_LIMIT:
        reasons.append(f"RAM {ram / GIB:.2f} GiB < 16 GiB")
    if vram is not None and vram < VRAM_LIMIT:
        reasons.append(f"{memory_type} {vram / GIB:.2f} GiB < 8 GiB")
    return bool(reasons), reasons, ram, vram
