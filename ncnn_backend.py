"""ncnn Vulkan inference using the existing DBV4 preprocessing and metadata."""

import importlib.util
import atexit
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import weakref
from typing import Sequence

import numpy as np
from PIL import Image

from dbv4 import DBV4Metadata, DBV4Preprocessor, infer_output_to_probabilities

_runtimes = weakref.WeakSet()


def _shutdown_vulkan(ncnn) -> None:
    # Windows can unload the Vulkan DLL before ncnn's native static destructor.
    # Release nets while the device still exists, then destroy the instance.
    for runtime in list(_runtimes):
        runtime.close()
    ncnn.destroy_gpu_instance()


def _register_vulkan_shutdown(ncnn) -> None:
    if (platform.system() == 'Windows'
            and callable(getattr(ncnn, 'destroy_gpu_instance', None))
            and not getattr(ncnn, '_dbv4_shutdown_registered', False)):
        atexit.register(_shutdown_vulkan, ncnn)
        ncnn._dbv4_shutdown_registered = True


def configure_vulkan_environment() -> None:
    """Apply the measured RADV synchronization workaround before instance creation."""
    if platform.system() != 'Linux':
        return
    for vendor in Path('/sys/class/drm').glob('card*/device/vendor'):
        try:
            is_amd = vendor.read_text().strip().lower() == '0x1002'
        except OSError:
            continue
        if is_amd:
            flags = [flag for flag in os.environ.get('RADV_DEBUG', '').split(',') if flag]
            if 'syncshaders' not in flags:
                flags.append('syncshaders')
                os.environ['RADV_DEBUG'] = ','.join(flags)
                print('[INFO] ncnn AMD RADV: shader同期を有効にします (syncshaders)。')
            return


def vulkan_device(ncnn, gpu_index: int) -> str:
    if not all(callable(getattr(ncnn, name, None)) for name in ('get_gpu_count', 'get_gpu_info')):
        raise RuntimeError('ncnn Python bindingにVulkan GPU APIがありません。Vulkan有効のビルドが必要です。')
    if gpu_index < 0:
        raise ValueError("GPU device index must be non-negative")
    _register_vulkan_shutdown(ncnn)
    count = ncnn.get_gpu_count()
    if gpu_index >= count:
        raise RuntimeError(f"ncnn Vulkan GPU {gpu_index} を利用できません (検出数: {count})")
    info = ncnn.get_gpu_info(gpu_index)
    name = info.device_name()
    # Reject ncnn CPU devices and known Mesa software implementations.
    if info.type() == 3 or any(part in name.lower() for part in ("llvmpipe", "softpipe", "lavapipe")):
        raise RuntimeError(f"ncnn Vulkan {gpu_index} はソフトウェアデバイスです: {name}")
    return name


def memory_budgets_mib(ncnn=None, gpu_index: int = 0) -> tuple[int | None, int | None]:
    """Read available RAM and the selected Vulkan device's heap budget."""
    ram = gpu = None
    try:
        if platform.system() == 'Linux':
            fields = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
            ram = int(fields['MemAvailable'].split()[0]) // 1024
        elif platform.system() == 'Windows':
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong) for name in (
                        'total_phys', 'avail_phys', 'total_page', 'avail_page',
                        'total_virtual', 'avail_virtual', 'avail_extended')]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = status.avail_phys // 1048576
        else:
            ram = os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') // 1048576
    except (OSError, ValueError, KeyError, AttributeError):
        pass
    if ncnn is not None:
        try:
            # Native ncnn reports this budget in MiB, for this device only.
            value = int(ncnn.get_gpu_device(gpu_index).get_heap_budget())
            gpu = value if value > 0 else None
        except (AttributeError, RuntimeError, TypeError, ValueError):
            pass
    return ram, gpu


def resolve_part_size_mib(device_name: str, requested: int | None,
                          model_prefix: Path | None = None, ncnn=None,
                          gpu_index: int = 0) -> int:
    """Choose streaming from memory headroom and actual FP32 weight size."""
    if requested is not None:
        if requested < 0:
            raise ValueError('--ncnn-part-size-mib must be nonnegative')
        return requested
    ram, gpu = memory_budgets_mib(ncnn, gpu_index)
    try:
        weights = model_prefix.with_suffix('.ncnn.bin').stat().st_size / 1048576 if model_prefix else None
    except OSError:
        weights = None
    if weights is not None:
        # Reserve room for activations, staging and CPU-side weight loading.
        ram_needed, gpu_needed = weights * 1.5 + 1024, weights * 1.25 + 256
        limited = [budget for budget, needed in ((ram, ram_needed), (gpu, gpu_needed))
                   if budget is not None and budget < needed]
        size = 0
        if limited:
            # Choose the largest power-of-two target that fits the measured
            # budget, rather than splitting every small-memory device at 128.
            targets = []
            if ram is not None:
                targets.append((ram - 1024) / 1.5)
            if gpu is not None:
                targets.append((gpu - 256) / 1.25)
            target = max(64, min(targets))
            size = 64
            while size * 2 <= target:
                size *= 2
        elif gpu is None and platform.system() == 'Linux' and any(
                part in device_name.lower() for part in ('liverpool', 'playstation 4', 'radv oberon')):
            size = 128  # Keep the tested PS4 fallback when Vulkan cannot report memory.
        # Liverpool can reset with large sections even when reported budgets fit.
        # Keep explicit requests above as overrides; retain 64 MiB for low budgets.
        if (weights > 128 and platform.system() == 'Linux' and any(
                part in device_name.lower() for part in
                ('liverpool', 'playstation 4', 'radv oberon'))):
            size = min(size or 128, 128)
            print('[INFO] ncnn PS4 RADV: GPUリセット回避のため自動分割を128 MiB以下に制限します。',
                  flush=True)
        print(f'[INFO] ncnn メモリ自動設定: 空きRAM={ram if ram is not None else "不明"} MiB, '
              f'GPUメモリ予算={gpu if gpu is not None else "不明"} MiB, 重み={weights:.0f} MiB; '
              f'分割目安={size} MiB (0=常駐)。', flush=True)
        return size
    if platform.system() == 'Linux' and any(
            part in device_name.lower() for part in ('liverpool', 'playstation 4', 'radv oberon')):
        print('[INFO] ncnn PS4 RADV: 省メモリ分割推論を自動選択します (128 MiB)。', flush=True)
        return 128
    return 0


def ensure_ncnn_model(model_prefix: Path, source_repo: str,
                      input_size: int, label_count: int) -> None:
    param = model_prefix.with_suffix(".ncnn.param")
    weights = model_prefix.with_suffix(".ncnn.bin")
    manifest = model_prefix.with_suffix(".ncnn.json")
    if param.is_file() and weights.is_file():
        if manifest.is_file():
            cached = json.loads(manifest.read_text(encoding="utf-8"))
            if (cached.get("source_repo") == source_repo
                    and cached.get("input_size") == input_size
                    and cached.get("label_count") == label_count
                    and cached.get("fp16") is False):
                return
    if not source_repo:
        raise FileNotFoundError(f"ncnnモデルがありません: {param}, {weights}")
    try:
        if (importlib.util.find_spec("torch") is None
                and platform.system() == "Linux"
                and platform.machine() == "x86_64"):
            print("[INFO] ncnn変換用のCPU版PyTorchを導入します。", flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
                            "--index-url", "https://download.pytorch.org/whl/cpu"], check=True)
        dependencies = [name for name in ("torch", "timm", "pnnx")
                        if importlib.util.find_spec(name) is None]
        if dependencies:
            print("[INFO] ncnn初回変換用ライブラリを導入します: " + ", ".join(dependencies), flush=True)
            subprocess.run([sys.executable, "-m", "pip", "install", *dependencies], check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "ncnn初回変換用ライブラリの導入に失敗しました。別のマシンで"
            "convert_ncnn_model.pyを実行し、model.ncnn.param/binを"
            f"{model_prefix.parent}へコピーできます: {exc}"
        ) from exc
    from convert_ncnn_model import convert

    try:
        convert(source_repo, model_prefix.parent, input_size, label_count)
    except Exception as exc:
        raise RuntimeError(
            f"ncnnモデル変換に失敗しました ({source_repo}): {exc}。"
            "Hugging Faceのアクセス権、空きメモリ、pnnxの演算対応を確認してください。"
        ) from exc


class NcnnRuntimeModel:
    """Expose the same prediction methods as the ONNX Runtime model."""

    batch_limit = 1  # pnnx exports a fixed batch-one graph.

    def __init__(self, metadata: DBV4Metadata, preprocessor: DBV4Preprocessor,
                 model_prefix: Path, gpu_index: int = 0,
                 precision: str = "fp32") -> None:
        configure_vulkan_environment()
        try:
            import ncnn
        except ImportError as exc:
            raise RuntimeError("ncnn Python bindingがありません。ncnnをインストールしてください。") from exc
        device_name = vulkan_device(ncnn, gpu_index)
        self.device_name = device_name
        binding = getattr(ncnn, 'ncnn', ncnn)
        self.binding_info = {
            'version': getattr(ncnn, '__version__', getattr(binding, '__version__', None)),
            'package_path': getattr(ncnn, '__file__', None),
            'binding_path': getattr(binding, '__file__', None),
        }
        binding_path = self.binding_info['binding_path']
        if binding_path and Path(binding_path).is_file():
            self.binding_info['sha256'] = hashlib.sha256(Path(binding_path).read_bytes()).hexdigest()
        param = model_prefix.with_suffix(".ncnn.param")
        weights = model_prefix.with_suffix(".ncnn.bin")
        missing = [str(path) for path in (param, weights) if not path.is_file()]
        if missing:
            raise FileNotFoundError("ncnnモデルがありません: " + ", ".join(missing))
        self.metadata = metadata
        self.preprocessor = preprocessor
        self.net = ncnn.Net()
        _runtimes.add(self)
        self.net.opt.use_vulkan_compute = True
        self.net.opt.use_fp16_storage = precision != "fp32"
        self.net.opt.use_fp16_packed = precision in ("fp16-packed", "fp16-arithmetic")
        self.net.opt.use_fp16_arithmetic = precision == "fp16-arithmetic"
        self.net.set_vulkan_device(gpu_index)
        if self.net.load_param(str(param)) != 0:
            raise RuntimeError(f"ncnn paramの読み込みに失敗しました: {param}")
        if self.net.load_model(str(weights)) != 0:
            raise RuntimeError(f"ncnn binの読み込みに失敗しました: {weights}")
        self.options = {name: bool(getattr(self.net.opt, name)) for name in (
            'use_vulkan_compute', 'use_fp16_storage', 'use_fp16_packed', 'use_fp16_arithmetic',
            'use_packing_layout', 'use_subgroup_ops', 'use_winograd_convolution',
            'use_sgemm_convolution', 'use_bf16_storage', 'use_bf16_packed',
            'use_shader_local_memory', 'use_local_pool_allocator')
            if hasattr(self.net.opt, name)}
        inputs = list(self.net.input_names())
        outputs = list(self.net.output_names())
        if len(inputs) != 1 or len(outputs) != 1:
            raise RuntimeError(f"ncnnモデルは単一入出力が必要です: inputs={inputs}, outputs={outputs}")
        self.input_name, self.output_name = inputs[0], outputs[0]
        print(f"[INFO] ncnn Vulkan GPU {gpu_index} ({device_name}): {metadata.profile_name}, "
              f"{precision}, ラベル数 {metadata.label_count}")

    def predict_images(self, images: Sequence[Image.Image]):
        return self.predict_preprocessed(
            np.stack([self.preprocessor(image) for image in images])
        ) if images else []

    def close(self) -> None:
        net = getattr(self, 'net', None)
        if net is not None:
            net.clear()
            self.net = None

    def predict_preprocessed(self, batch_nchw: np.ndarray):
        import ncnn

        results = []
        for sample in batch_nchw:
            # ncnn.Mat borrows NumPy storage; keep this array alive through extract.
            sample = np.ascontiguousarray(sample, dtype=np.float32)
            if sample.ndim != 3 or sample.shape[0] != 3:
                raise ValueError(f"ncnn入力はCHW 3チャネルが必要です: {sample.shape}")
            extractor = self.net.create_extractor()
            try:
                status = extractor.input(self.input_name, ncnn.Mat(sample))
                if status != 0:
                    raise RuntimeError(f"ncnn入力に失敗しました: status={status}")
                status, output = extractor.extract(self.output_name)
                if status != 0:
                    raise RuntimeError(f"ncnn推論に失敗しました: status={status}")
                # Retain only NumPy-owned output across validation errors.
                raw = np.asarray(output).reshape(-1).copy()
            finally:
                output = None
                extractor = None
            if raw.size != self.metadata.label_count:
                raise ValueError(
                    f"ncnn出力ラベル数が一致しません: {raw.size} != {self.metadata.label_count}"
                )
            results.append(infer_output_to_probabilities(raw))
        return results


class StreamingNcnnRuntimeModel(NcnnRuntimeModel):
    """Execute unchanged graph sections with one section resident at a time.

    This trades model reloads and CPU/GPU transfers for a lower memory peak.
    Memory-based selection or an explicit size enables this runtime.
    """

    def __init__(self, metadata, preprocessor, model_prefix, gpu_index=0,
                 precision='fp32', part_size_mib=128):
        from partition_ncnn_model import partition_model

        self._inference_lock = threading.Lock()
        self.part_directory, self.parts = partition_model(model_prefix, part_size_mib)
        self.gpu_index = gpu_index
        first = self.part_directory / self.parts[0]['param']
        super().__init__(metadata, preprocessor,
                         Path(str(first)[:-len('.ncnn.param')]), gpu_index, precision)
        self.close()
        print(f'[INFO] ncnn streaming: {len(self.parts)} sections, '
              f'largest {max(p["bytes"] for p in self.parts) / 1048576:.1f} MiB weights',
              flush=True)

    def predict_preprocessed(self, batch_nchw):
        # Server workers share this runtime. Reloading/clearing a section must
        # finish before another request can replace the same native net.
        with self._inference_lock:
            return self._predict_sections(batch_nchw)

    def _predict_sections(self, batch_nchw):
        import ncnn

        results = []
        for sample in batch_nchw:
            tensor = np.ascontiguousarray(sample, dtype=np.float32)
            if tensor.ndim != 3 or tensor.shape[0] != 3:
                raise ValueError(f'ncnn入力はCHW 3チャネルが必要です: {tensor.shape}')
            for index, part in enumerate(self.parts):
                print(f'[ncnn streaming] section {index + 1}/{len(self.parts)}', flush=True)
                self.net = ncnn.Net()
                extractor = output = mat = None
                try:
                    for name, value in self.options.items():
                        setattr(self.net.opt, name, value)
                    self.net.set_vulkan_device(self.gpu_index)
                    if self.net.load_param(str(self.part_directory / part['param'])) != 0:
                        raise RuntimeError(f'Cannot load ncnn section {index}')
                    if self.net.load_model(str(self.part_directory / part['bin'])) != 0:
                        raise RuntimeError(f'Cannot load ncnn section weights {index}')
                    extractor = self.net.create_extractor()
                    # Owned ncnn allocation also supplies native alignment.
                    mat = ncnn.Mat(tensor).clone()
                    if extractor.input(part['input'], mat) != 0:
                        raise RuntimeError(f'Cannot input ncnn section {index}')
                    status, output = extractor.extract(part['output'])
                    if status != 0:
                        raise RuntimeError(f'ncnn section {index} failed: {status}')
                    tensor = np.asarray(output).copy()
                finally:
                    output = extractor = mat = None
                    self.close()
            raw = tensor.reshape(-1)
            if raw.size != self.metadata.label_count:
                raise ValueError(f'ncnn label count mismatch: {raw.size} != {self.metadata.label_count}')
            results.append(infer_output_to_probabilities(raw))
        return results
