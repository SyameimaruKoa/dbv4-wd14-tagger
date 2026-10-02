"""ncnn Vulkan inference using the existing DBV4 preprocessing and metadata."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Sequence

import numpy as np
from PIL import Image

from dbv4 import DBV4Metadata, DBV4Preprocessor, infer_output_to_probabilities


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
    count = ncnn.get_gpu_count()
    if gpu_index >= count:
        raise RuntimeError(f"ncnn Vulkan GPU {gpu_index} を利用できません (検出数: {count})")
    info = ncnn.get_gpu_info(gpu_index)
    name = info.device_name()
    # VkPhysicalDeviceType 3 is a CPU device (e.g. Mesa llvmpipe).
    if info.type() == 3 or any(part in name.lower() for part in ("llvmpipe", "softpipe", "lavapipe")):
        raise RuntimeError(f"ncnn Vulkan {gpu_index} はソフトウェアデバイスです: {name}")
    return name


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

    def predict_preprocessed(self, batch_nchw: np.ndarray):
        import ncnn

        results = []
        for sample in batch_nchw:
            # ncnn.Mat borrows NumPy storage; keep this array alive through extract.
            sample = np.ascontiguousarray(sample, dtype=np.float32)
            if sample.ndim != 3 or sample.shape[0] != 3:
                raise ValueError(f"ncnn入力はCHW 3チャネルが必要です: {sample.shape}")
            extractor = self.net.create_extractor()
            status = extractor.input(self.input_name, ncnn.Mat(sample))
            if status != 0:
                raise RuntimeError(f"ncnn入力に失敗しました: status={status}")
            status, output = extractor.extract(self.output_name)
            if status != 0:
                raise RuntimeError(f"ncnn推論に失敗しました: status={status}")
            raw = np.asarray(output).reshape(-1)
            if raw.size != self.metadata.label_count:
                raise ValueError(
                    f"ncnn出力ラベル数が一致しません: {raw.size} != {self.metadata.label_count}"
                )
            results.append(infer_output_to_probabilities(raw))
        return results
