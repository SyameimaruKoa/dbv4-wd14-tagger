import json
import os
from pathlib import Path
import sys
import tempfile
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch

import numpy as np
from PIL import Image

from convert_ncnn_model import convert
from ncnn_backend import (NcnnRuntimeModel, StreamingNcnnRuntimeModel, configure_vulkan_environment,
                          ensure_ncnn_model, vulkan_device, _register_vulkan_shutdown,
                          resolve_part_size_mib)


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


class StreamingDefaultsTests(unittest.TestCase):
    def test_memory_and_weight_size_choose_resident_or_largest_fitting_part(self):
        prefix = SimpleNamespace(with_suffix=lambda suffix: SimpleNamespace(
            stat=lambda: SimpleNamespace(st_size=2700 * 1048576)))
        for ram, gpu, expected in ((16000, 8000, 0), (5000, 2048, 1024),
                                   (2000, 8000, 512), (16000, 1024, 512),
                                   (16000, 512, 128), (16000, None, 0),
                                   (7400, 4152, 0)):
            with self.subTest(ram=ram, gpu=gpu), \
                    patch('ncnn_backend.memory_budgets_mib', return_value=(ram, gpu)), \
                    patch('ncnn_backend.platform.system', return_value='Linux'):
                self.assertEqual(resolve_part_size_mib('test GPU', None, prefix), expected)

    def test_explicit_size_does_not_probe_memory(self):
        with patch('ncnn_backend.memory_budgets_mib') as probe:
            self.assertEqual(resolve_part_size_mib('test GPU', 0), 0)
            self.assertEqual(resolve_part_size_mib('test GPU', 128), 128)
            probe.assert_not_called()

    def test_memory_probe_uses_selected_device_and_mib(self):
        from ncnn_backend import memory_budgets_mib
        from unittest.mock import Mock
        get_device = Mock(return_value=SimpleNamespace(get_heap_budget=lambda: 2048))
        with patch('ncnn_backend.platform.system', return_value='Linux'), \
                patch('ncnn_backend.Path.read_text', return_value='MemTotal: 8192000 kB\nMemAvailable: 4096000 kB\n'):
            self.assertEqual(memory_budgets_mib(SimpleNamespace(get_gpu_device=get_device), 2), (4000, 2048))
        get_device.assert_called_once_with(2)

    def test_unavailable_memory_probe_does_not_break_startup(self):
        from ncnn_backend import memory_budgets_mib
        with patch('ncnn_backend.platform.system', return_value='Linux'), \
                patch('ncnn_backend.Path.read_text', side_effect=OSError):
            self.assertEqual(memory_budgets_mib(SimpleNamespace()), (None, None))

    def test_runtime_selects_streaming_before_startup_probe(self):
        import embed_tags_universal as app

        for device, requested, expected in (
                ('AMD Liverpool (PlayStation 4) (RADV LIVERPOOL)', None, 128),
                ('AMD Liverpool (PlayStation 4) (RADV LIVERPOOL)', 0, 0),
                ('AMD Radeon Graphics (RADV RENOIR)', None, 0)):
            with self.subTest(device=device, requested=requested), \
                    patch('ncnn_backend.platform.system', return_value='Linux'), \
                    patch('ncnn_backend.configure_vulkan_environment'), \
                    patch('ncnn_backend.vulkan_device', return_value=device), \
                    patch.dict(sys.modules, {'ncnn': SimpleNamespace()}), \
                    patch.object(app, 'ensure_profile_access'), \
                    patch.object(app.DBV4Metadata, 'load'), \
                    patch.object(app.DBV4Preprocessor, 'from_metadata'), \
                    patch('ncnn_backend.NcnnRuntimeModel') as resident, \
                    patch('ncnn_backend.StreamingNcnnRuntimeModel') as streaming:
                runtime = app.load_runtime_model(
                    True, 'ultra', provider='ncnn', model_file='/tmp/model',
                    ncnn_part_size_mib=requested)
                selected, other = (streaming, resident) if expected else (resident, streaming)
                other.assert_not_called()
                selected.assert_called_once()
                if expected:
                    self.assertEqual(selected.call_args.args[-1], expected)
                self.assertIs(runtime, selected.return_value)
                runtime.predict_images.assert_called_once()

    def test_ps4_linux_defaults_to_tested_streaming_size(self):
        with patch('ncnn_backend.platform.system', return_value='Linux'):
            for name in ('AMD Liverpool (PlayStation 4) (RADV LIVERPOOL)',
                         'AMD Radeon Graphics (RADV OBERON)'):
                with self.subTest(name=name):
                    self.assertEqual(resolve_part_size_mib(name, None), 128)

    def test_other_devices_keep_resident_inference(self):
        for system, name in (('Linux', 'AMD Radeon Graphics (RADV RENOIR)'),
                             ('Linux', 'NVIDIA Tegra'), ('Windows', 'AMD Liverpool')):
            with self.subTest(system=system, name=name), \
                    patch('ncnn_backend.platform.system', return_value=system):
                self.assertEqual(resolve_part_size_mib(name, None), 0)

    def test_explicit_zero_and_size_override_ps4_default(self):
        with patch('ncnn_backend.platform.system', return_value='Linux'):
            for requested in (0, 64, 256):
                self.assertEqual(resolve_part_size_mib('RADV LIVERPOOL', requested), requested)
            with self.assertRaisesRegex(ValueError, 'nonnegative'):
                resolve_part_size_mib('RADV LIVERPOOL', -1)


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

    def test_streaming_requests_do_not_clear_another_requests_net(self):
        entered = threading.Event()
        release = threading.Event()
        second_started = threading.Event()
        second_net_created = threading.Event()
        nets = []

        class Mat:
            def __init__(self, value):
                self.value = value

            def clone(self):
                return self.value.copy()

        class Net(FakeNet):
            def __init__(self):
                super().__init__()
                self.cleared = False
                self.index = len(nets)
                nets.append(self)
                if self.index == 1:
                    second_net_created.set()

            def clear(self):
                self.cleared = True

            def create_extractor(self):
                net = self

                class Extractor(FakeExtractor):
                    def extract(self, name):
                        if net.index == 0:
                            entered.set()
                            if not release.wait(3):
                                raise RuntimeError('test inference timed out')
                        if net.cleared:
                            raise RuntimeError('active request net was cleared')
                        return 0, self.input_tensor[1].reshape(-1)[:4].copy()

                return Extractor(None)

        runtime = StreamingNcnnRuntimeModel.__new__(StreamingNcnnRuntimeModel)
        runtime._inference_lock = threading.Lock()
        runtime.part_directory = Path('.')
        runtime.parts = [{'param': 'part.param', 'bin': 'part.bin',
                          'input': 'in', 'output': 'out'}]
        runtime.options = {}
        runtime.gpu_index = 0
        runtime.metadata = SimpleNamespace(label_count=4)
        runtime.net = None
        binding = SimpleNamespace(Net=Net, Mat=Mat)

        def second_request():
            second_started.set()
            return runtime.predict_preprocessed(np.full((1, 3, 2, 2), 0.8, np.float32))

        with patch.dict(sys.modules, {'ncnn': binding}), ThreadPoolExecutor(2) as pool:
            first = pool.submit(runtime.predict_preprocessed,
                                np.full((1, 3, 2, 2), 0.2, np.float32))
            try:
                self.assertTrue(entered.wait(3))
                second = pool.submit(second_request)
                self.assertTrue(second_started.wait(3))
                self.assertFalse(second_net_created.wait(0.1))
                self.assertFalse(second.done())
                self.assertEqual(len(nets), 1)
            finally:
                release.set()
            np.testing.assert_allclose(first.result(timeout=3)[0], [0.2] * 4)
            np.testing.assert_allclose(second.result(timeout=3)[0], [0.8] * 4)
        self.assertEqual(len(nets), 2)
        self.assertTrue(all(net.cleared for net in nets))

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
