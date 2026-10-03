import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from PIL import Image

from convert_ncnn_model import convert
from ncnn_backend import (NcnnRuntimeModel, configure_vulkan_environment,
                          ensure_ncnn_model, vulkan_device, _register_vulkan_shutdown)


class VulkanShutdownTests(unittest.TestCase):
    def test_windows_shutdown_releases_models_before_instance_once(self):
        events = []
        binding = SimpleNamespace(destroy_gpu_instance=lambda: events.append('instance'))
        runtime = SimpleNamespace(close=lambda: events.append('model'))
        with patch('ncnn_backend.platform.system', return_value='Windows'), \
                patch('ncnn_backend.atexit.register') as register, \
                patch('ncnn_backend._runtimes', [runtime]):
            _register_vulkan_shutdown(binding)
            _register_vulkan_shutdown(binding)
            register.assert_called_once()
            callback, module = register.call_args.args
            callback(module)
        self.assertEqual(events, ['model', 'instance'])

    def test_close_releases_net_and_can_be_repeated(self):
        events = []
        runtime = NcnnRuntimeModel.__new__(NcnnRuntimeModel)
        runtime.net = SimpleNamespace(clear=lambda: events.append('clear'))
        runtime.close()
        runtime.close()
        self.assertIsNone(runtime.net)
        self.assertEqual(events, ['clear'])


class VulkanEnvironmentTests(unittest.TestCase):
    def test_amd_sync_preserves_existing_flags_and_is_idempotent(self):
        vendor = SimpleNamespace(read_text=lambda: '0x1002\n')
        with patch('ncnn_backend.platform.system', return_value='Linux'), \
                patch('ncnn_backend.Path.glob', return_value=[vendor]), \
                patch.dict(os.environ, {'RADV_DEBUG': 'nocache'}, clear=True):
            configure_vulkan_environment()
            configure_vulkan_environment()
            self.assertEqual(os.environ['RADV_DEBUG'], 'nocache,syncshaders')

    def test_other_vendor_keeps_driver_environment(self):
        vendor = SimpleNamespace(read_text=lambda: '0x8086\n')
        with patch('ncnn_backend.platform.system', return_value='Linux'), \
                patch('ncnn_backend.Path.glob', return_value=[vendor]), \
                patch.dict(os.environ, {}, clear=True):
            configure_vulkan_environment()
            self.assertNotIn('RADV_DEBUG', os.environ)


class FakeExtractor:
    def __init__(self, output):
        self.output = output
        self.input_tensor = None

    def input(self, name, mat):
        self.input_tensor = (name, np.asarray(mat))
        return 0

    def extract(self, name):
        return 0, self.output


class FakeNet:
    def __init__(self):
        self.opt = SimpleNamespace()
        self.extractors = []
        self.device = None

    def set_vulkan_device(self, index):
        self.device = index

    def load_param(self, path):
        return 0

    def load_model(self, path):
        return 0

    def input_names(self):
        return ["in0"]

    def output_names(self):
        return ["out0"]

    def create_extractor(self):
        result = FakeExtractor(np.array([0.1, 0.4, 0.8, 0.9], np.float32))
        self.extractors.append(result)
        return result


class NcnnBackendTests(unittest.TestCase):
    def test_runtime_returns_dbv4_probabilities_for_each_image(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "model"
            prefix.with_suffix(".ncnn.param").write_text("param")
            prefix.with_suffix(".ncnn.bin").write_bytes(b"bin")
            net = FakeNet()
            fake_ncnn = SimpleNamespace(Net=lambda: net, Mat=lambda value: value,
                                        get_gpu_count=lambda: 1,
                                        get_gpu_info=lambda index: SimpleNamespace(
                                            device_name=lambda: "test GPU", type=lambda: 2))
            metadata = SimpleNamespace(label_count=4, profile_name="ultra")
            preprocessor = lambda image: np.ones((3, 4, 4), np.float32)
            with patch.dict(sys.modules, {"ncnn": fake_ncnn}):
                runtime = NcnnRuntimeModel(metadata, preprocessor, prefix,
                                           precision="fp16-packed")
                result = runtime.predict_images([Image.new("RGB", (4, 4))] * 2)
            self.assertEqual(runtime.batch_limit, 1)
            self.assertEqual(len(result), 2)
            np.testing.assert_allclose(result[0], [0.1, 0.4, 0.8, 0.9])
            self.assertEqual(net.extractors[0].input_tensor[1].shape, (3, 4, 4))
            self.assertTrue(net.opt.use_vulkan_compute)
            self.assertTrue(net.opt.use_fp16_storage)
            self.assertTrue(net.opt.use_fp16_packed)
            self.assertFalse(net.opt.use_fp16_arithmetic)

    def test_software_vulkan_device_is_rejected(self):
        ncnn = SimpleNamespace(get_gpu_count=lambda: 1,
                               get_gpu_info=lambda index: SimpleNamespace(
                                   device_name=lambda: "llvmpipe", type=lambda: 3))
        with self.assertRaisesRegex(RuntimeError, "ソフトウェアデバイス"):
            vulkan_device(ncnn, 0)

    def test_borrowed_input_storage_survives_until_extraction(self):
        class BorrowingExtractor:
            def input(self, name, sample_reference):
                self.sample_reference = sample_reference
                return 0

            def extract(self, name):
                sample = self.sample_reference()
                if sample is None:
                    raise RuntimeError('ncnn input storage was released')
                return 0, np.array([0.1, 0.4, 0.8, 0.9], np.float32)

        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / 'model'
            prefix.with_suffix('.ncnn.param').write_text('param')
            prefix.with_suffix('.ncnn.bin').write_bytes(b'bin')
            net = FakeNet()
            net.create_extractor = BorrowingExtractor
            binding = SimpleNamespace(Net=lambda: net, Mat=weakref.ref,
                                      get_gpu_count=lambda: 1,
                                      get_gpu_info=lambda index: SimpleNamespace(
                                          device_name=lambda: 'test GPU', type=lambda: 2))
            metadata = SimpleNamespace(label_count=4, profile_name='ultra')
            with patch.dict(sys.modules, {'ncnn': binding}):
                runtime = NcnnRuntimeModel(metadata, None, prefix)
                # Float16 forces allocation of new float32 input storage.
                result = runtime.predict_preprocessed(np.zeros((2, 3, 4, 4), np.float16))
            self.assertEqual(len(result), 2)

    def test_cpu_only_binding_reports_missing_vulkan_support(self):
        with self.assertRaisesRegex(RuntimeError, 'Vulkan GPU API'):
            vulkan_device(SimpleNamespace(), 0)

    def test_cached_converter_files_do_not_require_conversion_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            (dest / "model.ncnn.param").write_text("param")
            (dest / "model.ncnn.bin").write_bytes(b"bin")
            (dest / "model.ncnn.json").write_text(json.dumps({
                "source_repo": "owner/model", "input_size": 512,
                "label_count": 4, "fp16": False,
            }))
            ensure_ncnn_model(dest / "model", "owner/model", 512, 4)
            self.assertEqual(convert("owner/model", dest, 512, 4),
                             (dest / "model.ncnn.param", dest / "model.ncnn.bin"))

    def test_killed_converter_reports_memory_failure_without_publishing_cache(self):
        for exit_code in (-9, 247):
            with self.subTest(exit_code=exit_code), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / ('pnnx.exe' if os.name == 'nt' else 'pnnx')).write_text('stub')
                destination = root / 'models'
                with patch.object(sys, 'executable', str(root / 'python')), patch(
                    'convert_ncnn_model.subprocess.run', side_effect=[
                        subprocess.CompletedProcess([], 0),
                        subprocess.CalledProcessError(exit_code, 'pnnx'),
                    ]
                ) as run:
                    with self.assertRaisesRegex(RuntimeError, 'RAM/swap'):
                        convert('owner/model', destination, 8, 4)
                self.assertIn('--trace-only', run.call_args_list[0].args[0])
                self.assertFalse((destination / 'model.ncnn.json').exists())
                self.assertEqual(list(destination.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
