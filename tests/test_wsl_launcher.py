"""Check WSL invocation without starting containers or downloading models."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
POWERSHELL = shutil.which('powershell') or shutil.which('pwsh')


@unittest.skipUnless(POWERSHELL, 'PowerShell required')
class WslLauncherTests(unittest.TestCase):
    def launch(self, arguments, status=0):
        with tempfile.TemporaryDirectory(prefix='dbv4 WSL 日本語 ') as folder:
            trace = Path(folder) / 'trace.json'
            script = Path(folder) / 'invoke.ps1'
            # Function mocking keeps argument boundaries and propagates exit codes.
            script.write_text(
                "function global:wslc {\n"
                "    ConvertTo-Json -InputObject @($args) -Compress | "
                "Set-Content -LiteralPath $env:WSL_TEST_TRACE -Encoding UTF8\n"
                "    $global:LASTEXITCODE = [int]$env:WSL_TEST_STATUS\n"
                "}\n"
                "try { & $env:WSL_TEST_LAUNCHER " + arguments
                + ' } catch { Write-Error $_; exit 1 }\nexit $LASTEXITCODE\n',
                encoding='utf-8-sig',
            )
            result = subprocess.run(
                [POWERSHELL, '-NoProfile', '-File', str(script)],
                capture_output=True, env={**os.environ,
                    'WSL_TEST_TRACE': str(trace), 'WSL_TEST_STATUS': str(status),
                    'WSL_TEST_LAUNCHER': str(ROOT / 'run_tagger_wsl.ps1'),
                    'WSL_TEST_IMAGES': folder},
            )
            invocation = json.loads(trace.read_text(encoding='utf-8-sig')) if trace.exists() else None
            return result, invocation

    def test_server_exposes_all_gpus_and_selected_port(self):
        result, args = self.launch('-Mode Server -Provider intel -GpuIndex 2 -Port 5100 -PublishAddress 0.0.0.0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index('--gpus') + 1], 'all')
        self.assertEqual(args[args.index('--publish') + 1], '0.0.0.0:5100:5100')
        self.assertEqual(args[args.index('--gpu-index') + 1], '2')
        self.assertIn('type=volume,source=dbv4-tagger-wsl-workspace,target=/workspace', args)

    def test_image_mount_preserves_spaces_unicode_and_extra_arguments(self):
        result, args = self.launch("-Mode Standalone -Provider cpu -Path $env:WSL_TEST_IMAGES -TaggerArgs @('--recursive', '--no-report')")
        self.assertEqual(result.returncode, 0, result.stderr)
        mount = next(arg for arg in args if arg.startswith('type=bind,'))
        self.assertIn('dbv4 WSL 日本語 ', mount)
        self.assertTrue(mount.endswith(',target=/images'))
        self.assertEqual(args[-2:], ['--recursive', '--no-report'])
        self.assertNotIn('--gpus', args)
        self.assertNotIn('--publish', args)

    def test_client_uses_remote_host_without_gpu(self):
        result, args = self.launch('-Mode Client -Provider cuda -Path $env:WSL_TEST_IMAGES -HostName 192.0.2.1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index('--host') + 1], '192.0.2.1')
        self.assertNotIn('--gpus', args)

    def test_shared_data_and_vendor_libraries_use_separate_mounts(self):
        result, args = self.launch('-Mode Probe -Provider rocm -DataPath $env:WSL_TEST_IMAGES -GpuRuntimePath $env:WSL_TEST_IMAGES')
        self.assertEqual(result.returncode, 0, result.stderr)
        mounts = [arg for arg in args if arg.startswith('type=bind,')]
        self.assertTrue(any(arg.endswith(',target=/workspace/.dbv4') for arg in mounts))
        self.assertTrue(any(arg.endswith(',target=/gpu-runtime,readonly') for arg in mounts))
        self.assertIn('all', args)

    def test_file_input_maps_to_parent_and_container_filename(self):
        result, args = self.launch("-Mode Standalone -Path (Join-Path $env:WSL_TEST_IMAGES 'invoke.ps1')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index('--path') + 1], '/images/invoke.ps1')

    def test_build_passes_base_image_without_shell_interpolation(self):
        result, args = self.launch('-Action Build -BaseImage vendor/runtime:version -Image dbv4:test')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[0], 'build')
        self.assertEqual(args[args.index('--build-arg') + 1], 'BASE_IMAGE=vendor/runtime:version')
        self.assertEqual(args[args.index('--tag') + 1], 'dbv4:test')

    def test_exit_status_propagates(self):
        for mode in ['-Mode Probe -Provider cuda', '-Action Build']:
            with self.subTest(mode=mode):
                result, args = self.launch(mode, status=7)
                self.assertEqual(result.returncode, 7, result.stderr)
                self.assertIsNotNone(args)

    def test_missing_inputs_stop_before_run(self):
        for arguments in ['-Mode Standalone', '-Mode Client -Path $env:WSL_TEST_IMAGES']:
            with self.subTest(arguments=arguments):
                result, args = self.launch(arguments)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(args)

    def test_help_never_invokes_wslc(self):
        for arguments in ['', '-h', '-Action Help']:
            with self.subTest(arguments=arguments):
                result, args = self.launch(arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIsNone(args)
                self.assertIn(b'WorkspaceVolume', result.stdout)

    def test_powershell_source_has_utf8_bom(self):
        self.assertTrue((ROOT / 'run_tagger_wsl.ps1').read_bytes().startswith(b'\xef\xbb\xbf'))


if __name__ == '__main__':
    unittest.main()
