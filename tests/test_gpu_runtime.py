"""Regressions found during the September GPU benchmark runs (no hardware needed)."""
import json
import os
import subprocess
import shutil
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import gpu_runtime as gpu
import embed_tags_universal as app


def adapter(vendor_id, dxgi='0'):
    return SimpleNamespace(device=SimpleNamespace(
        vendor='', vendor_id=vendor_id, device_id=1,
        metadata={'DxgiAdapterNumber': dxgi}))


class GPUInitializationTests(unittest.TestCase):
    def test_webgpu_selects_only_requested_device_with_numeric_vendor(self):
        devices = [adapter(0x10DE), adapter(0x8086, '3')]
        options = MagicMock()
        with patch.object(gpu, 'webgpu_devices', return_value=devices):
            result = gpu.configure_webgpu(options, 1, 'intel')
        options.add_provider_for_devices.assert_called_once_with([devices[1]], {})
        self.assertEqual(result['vendor'], 'intel')

    def test_webgpu_rejects_wrong_vendor_invalid_index_and_ambiguity(self):
        with patch.object(gpu, 'webgpu_devices', return_value=[adapter(0x10DE), adapter(0x8086)]):
            for index, vendor in [(0, 'intel'), (-1, None), (2, None), (None, None)]:
                with self.subTest(index=index, vendor=vendor), self.assertRaises(RuntimeError):
                    gpu.configure_webgpu(MagicMock(), index, vendor)

    def test_webgpu_auto_selects_unique_vendor_match(self):
        devices = [adapter(0x10DE), adapter(0x8086)]
        with patch.object(gpu, 'webgpu_devices', return_value=devices):
            options = MagicMock()
            gpu.configure_webgpu(options, None, 'intel')
            options.add_provider_for_devices.assert_called_once_with([devices[1]], {})

    def test_webgpu_auto_selects_first_duplicate_vendor_match(self):
        devices = [adapter(0x8086, '1'), adapter(0x8086, '3'), adapter(0x10DE, '0')]
        with patch.object(gpu, 'webgpu_devices', return_value=devices):
            options = MagicMock()
            gpu.configure_webgpu(options, None, 'intel')
            options.add_provider_for_devices.assert_called_once_with([devices[0]], {})

    def test_directml_checks_dxgi_index_not_webgpu_index(self):
        records = [dict(vendor='', vendor_id=0x8086, metadata={'DxgiAdapterNumber':'3'}),
                   dict(vendor='', vendor_id=0x10DE, metadata={'DxgiAdapterNumber':'0'})]
        with patch.object(Path, 'is_file', return_value=True), patch.object(
            gpu.subprocess, 'check_output', return_value=json.dumps(records)
        ) as probe:
            gpu.validate_directml(3, 'intel')
            self.assertIn('venv_webgpu', probe.call_args.args[0][0])
            with self.assertRaises(RuntimeError):
                gpu.validate_directml(0, 'intel')
            with self.assertRaises(RuntimeError):
                gpu.validate_directml(1, 'intel')

    def test_openvino_rejects_nvidia_gpu_zero(self):
        with patch.object(gpu.platform, 'system', return_value='Linux'), patch.object(
            gpu.subprocess, 'check_output', return_value='NVIDIA RTX 2070\n'
        ), self.assertRaisesRegex(RuntimeError, 'Intel GPU'):
            gpu.prepare_openvino('GPU.0')

    def test_openvino_queries_in_subprocess(self):
        with patch.object(gpu.platform, 'system', return_value='Linux'), patch.object(
            gpu.subprocess, 'check_output', return_value='Intel UHD 630\n'
        ) as query:
            self.assertEqual(gpu.prepare_openvino('GPU.1'), 'Intel UHD 630')
            self.assertEqual(query.call_args.args[0][-1], 'GPU.1')
            self.assertIn('import openvino', query.call_args.args[0][2])
        with self.assertRaises(RuntimeError):
            gpu.prepare_openvino('CPU')

    def test_directml_caformer_workaround(self):
        options = gpu.ort.SessionOptions()
        gpu.configure_directml(options)
        self.assertFalse(options.enable_mem_pattern)
        self.assertEqual(options.execution_mode, gpu.ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(options.graph_optimization_level, gpu.ort.GraphOptimizationLevel.ORT_DISABLE_ALL)

    def test_windows_cudnn_preloads_tensor_ir_and_keeps_handles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'onnxruntime').mkdir()
            libs = root/'nvidia'/'cudnn'/'bin'; libs.mkdir(parents=True)
            (libs/'cudnn_engines_tensor_ir64_9.dll').touch()
            with patch.object(gpu.ort, '__file__', str(root/'onnxruntime'/'__init__.py')), patch.object(
                gpu.platform, 'system', return_value='Windows'
            ), patch.object(gpu.ort, 'preload_dlls') as preload, patch.object(
                gpu.os, 'add_dll_directory', create=True
            ) as dll_dir, patch.object(gpu.ctypes, 'WinDLL', create=True) as dll, patch.dict(os.environ):
                handles = gpu.prepare_cuda()
                preload.assert_called_once()
                dll.assert_called_once_with(str(libs/'cudnn_engines_tensor_ir64_9.dll'))
                self.assertIn(dll_dir.return_value, handles)
                self.assertIn(dll.return_value, gpu._RUNTIME_HANDLES)

    def test_tensorrt_missing_explicit_dir_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, 'selected directory'):
                gpu.prepare_tensorrt_linux(directory)

    @unittest.skipUnless(hasattr(gpu.ctypes, "RTLD_GLOBAL"), "Linux shared-library loading")
    def test_tensorrt_loads_plugin_and_ort_bridge(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            root = Path(directory)
            for name in ['libnvinfer.so.10', 'libnvinfer_plugin.so.10']:
                (root/name).touch()
            with patch.object(gpu.ctypes, 'CDLL') as load:
                found, handles = gpu.prepare_tensorrt_linux(root)
                self.assertEqual(found, root)
                self.assertEqual(len(handles), 3)
                self.assertEqual(load.call_count, 3)
                self.assertTrue(load.call_args.args[0].endswith('libonnxruntime_providers_tensorrt.so'))

    def test_explicit_tensorrt_failure_is_not_cuda_success(self):
        with patch.object(app.ort, 'get_available_providers', return_value=['TensorrtExecutionProvider','CUDAExecutionProvider']), patch.object(
            gpu, 'prepare_tensorrt', side_effect=RuntimeError('missing runtime')
        ), self.assertRaisesRegex(RuntimeError, 'missing runtime'):
            app.build_providers(True, provider='tensorrt')

    def test_explicit_missing_migraphx_is_not_cpu_success(self):
        with patch.object(app.ort, 'get_available_providers', return_value=['CPUExecutionProvider']), self.assertRaisesRegex(RuntimeError, 'unavailable'):
            app.build_providers(True, provider='migraphx')

    def test_provider_indices_and_cpu_override(self):
        with patch.object(app.ort, 'get_available_providers', return_value=['CUDAExecutionProvider']), patch.object(gpu, 'prepare_cuda'):
            self.assertEqual(app.build_providers(True, provider='cuda', gpu_index=2),
                             [('CUDAExecutionProvider', {'device_id': '2'}), 'CPUExecutionProvider'])
        self.assertEqual(app.build_providers(True, provider='cpu'), ['CPUExecutionProvider'])

    def test_cli_options_reach_both_normal_and_server_loader(self):
        args = app.create_parser().parse_args(['--provider','tensorrt','--gpu-index','2','--tensorrt-lib-dir','/tmp/trt'])
        for entry in (app.run_server, app.process_images):
            with self.subTest(entry=entry.__name__), patch.object(
                app, 'load_runtime_model', side_effect=RuntimeError('stop before inference')
            ) as load:
                with self.assertRaisesRegex(RuntimeError, 'stop before inference'):
                    entry(args)
                self.assertEqual(load.call_args.kwargs['provider'], 'tensorrt')
                self.assertEqual(load.call_args.kwargs['gpu_index'], 2)
                self.assertEqual(load.call_args.kwargs['tensorrt_lib_dir'], '/tmp/trt')

    def test_node_profile_does_not_count_active_provider_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'profile.json'
            path.write_text(json.dumps([
                {'cat':'Session', 'args':{'provider':'CUDAExecutionProvider'}},
                {'cat':'Node', 'args':{'provider':'CPUExecutionProvider'}},
            ]))
            self.assertEqual(gpu.executed_providers(path), ['CPUExecutionProvider'])

    def test_windows_tensorrt_detects_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            dirs = [Path(directory)/'one', Path(directory)/'two']
            for d in dirs:
                d.mkdir(); (d/'nvinfer_10.dll').touch()
            with patch.dict(os.environ, {'PATH':os.pathsep.join(map(str, dirs))}, clear=True), patch.object(
                gpu, 'prepare_cuda'
            ), patch.object(Path, 'home', return_value=Path(directory)):
                with self.assertRaisesRegex(RuntimeError, 'Multiple TensorRT'):
                    gpu.prepare_tensorrt_windows(None)

    def test_windows_tensorrt_discovers_runtime_in_portable_venv(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'PATH': ''}, clear=True):
            root = Path(directory)
            runtime = root/'Lib'/'site-packages'/'tensorrt_cu12_libs'
            runtime.mkdir(parents=True)
            (runtime/'nvinfer_10.dll').touch()
            (root/'onnxruntime'/'capi').mkdir(parents=True)
            with patch.object(gpu.sys, 'prefix', str(root)), patch.object(
                gpu.ort, '__file__', str(root/'onnxruntime'/'__init__.py')
            ), patch.object(gpu, 'prepare_cuda'), patch.object(
                gpu.os, 'add_dll_directory', create=True
            ), patch.object(gpu.ctypes, 'WinDLL', create=True):
                found, _ = gpu.prepare_tensorrt_windows(None)
                self.assertEqual(found, runtime)

    def test_tensorrt_provider_options_use_portable_cache(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(
            os.environ, {'TENSORRT_ENGINE_CACHE_DIR': directory}, clear=True
        ):
            options = gpu.tensorrt_provider_options(2)
            self.assertEqual(options['device_id'], '2')
            self.assertEqual(options['trt_engine_cache_enable'], 'True')
            self.assertEqual(Path(options['trt_engine_cache_path']), Path(directory).resolve())

    def test_windows_openvino_keeps_dll_search_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'onnxruntime').mkdir()
            libs = root/'openvino'/'libs'; libs.mkdir(parents=True)
            (libs/'openvino.dll').touch()
            with patch.object(gpu.ort, '__file__', str(root/'onnxruntime'/'__init__.py')), patch.object(
                gpu.platform, 'system', return_value='Windows'
            ), patch.object(gpu.os, 'add_dll_directory', create=True) as dll_dir, patch.object(
                gpu.ctypes, 'WinDLL', create=True
            ) as dll, patch.object(gpu.subprocess, 'check_output', return_value='Intel Xe'), patch.dict(os.environ):
                self.assertEqual(gpu.prepare_openvino('GPU.0'), 'Intel Xe')
                dll.assert_called_once_with(str(libs/'openvino.dll'))
                self.assertIn(dll_dir.return_value, gpu._RUNTIME_HANDLES)

    @unittest.skipUnless(os.name == "posix" and shutil.which("bash") and Path("/usr/bin/python3").is_file(), "POSIX Bash launcher")
    def test_bash_launcher_forwards_provider_indices_and_exit_status(self):
        source = Path(app.SCRIPT_DIR)/'run_tagger.sh'
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            launcher = root/'run_tagger.sh'; launcher.write_bytes(source.read_bytes())
            log = root/'args.json'
            python_stub = "#!/usr/bin/python3\nimport sys, json, os\nif sys.argv[1] == '-c':\n print('13' if 'version_info.minor' in sys.argv[2] else '')\nelse:\n json.dump(sys.argv[1:], open(os.environ['GPU_TEST_ARGS'], 'w')); sys.exit(7)\n"
            for name in ['venv_gpu', 'venv_webgpu', 'venv_intel', 'venv_std']:
                bindir = root/name/'bin'; bindir.mkdir(parents=True)
                for name2, content in [('python', python_stub), ('pip', '#!/bin/sh\nexit 0\n')]:
                    exe = bindir/name2; exe.write_text(content); exe.chmod(0o755)
            env = dict(os.environ, GPU_TEST_ARGS=str(log), HF_HOME=str(root/'hf'))
            for provider, options in [('cuda', ['--gpu-index','2']), ('intel',['--openvino-device','GPU.1']),
                                      ('webgpu',['--webgpu-device-index','1','--target-vendor','intel'])]:
                with self.subTest(provider=provider):
                    result = subprocess.run(['bash', str(launcher), '--provider', provider, *options, '/tmp/images with spaces'],
                                            env=env, capture_output=True, text=True, timeout=20)
                    self.assertEqual(result.returncode, 7, result.stdout+result.stderr)
                    args = json.loads(log.read_text())
                    self.assertEqual(args[args.index('--provider')+1], provider)
                    for option in options:
                        self.assertIn(option, args)
                    self.assertIn('/tmp/images with spaces', args)

    def _load_with_profile(self, executed, active=None):
        """Exercise the real loader, using an ORT session double at the boundary."""
        metadata = SimpleNamespace(family='dbv4', model_path='model.onnx', repo_id='test/model',
                                   profile_name='balanced', label_count=5, metadata_version='test')
        session = MagicMock()
        session.get_providers.return_value = active or ['CUDAExecutionProvider','CPUExecutionProvider']
        session.get_inputs.return_value = [SimpleNamespace(shape=['batch',3,384,384], name='input')]
        runtime = MagicMock(batch_limit=None)
        with patch.object(app, 'ensure_profile_access'), patch.object(app.DBV4Metadata, 'load', return_value=metadata), patch.object(
            app.DBV4Preprocessor, 'from_metadata'
        ), patch.object(app, 'build_providers', return_value=[('CUDAExecutionProvider', {'device_id':'0'}), 'CPUExecutionProvider']), patch.object(
            app.ort, 'InferenceSession', return_value=session
        ), patch.object(app, 'RuntimeModel', return_value=runtime), patch.object(gpu, 'executed_providers', return_value=executed):
            result = app.load_runtime_model(True, 'balanced', provider='cuda')
        session.disable_fallback.assert_called_once()
        session.end_profiling.assert_called_once()
        runtime.predict_images.assert_called_once()
        return result

    def test_loader_accepts_verified_gpu_nodes(self):
        self._load_with_profile(['CUDAExecutionProvider','CPUExecutionProvider'])

    def test_loader_rejects_cpu_only_execution_even_if_gpu_active(self):
        with self.assertRaisesRegex(RuntimeError, 'executed no nodes'):
            self._load_with_profile(['CPUExecutionProvider'])

    def test_loader_rejects_session_creation_cpu_fallback(self):
        with self.assertRaisesRegex(RuntimeError, 'unavailable'):
            self._load_with_profile(['CPUExecutionProvider'], ['CPUExecutionProvider'])


if __name__ == '__main__':
    unittest.main()
