"""Verify shared WSL dispatch without touching native environments."""
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
    def launch(self, arguments='', missing=0, status=0):
        with tempfile.TemporaryDirectory(prefix='WSL 日本語 ') as folder:
            trace = Path(folder) / 'trace.json'
            script = Path(folder) / 'invoke.ps1'
            script.write_text(
                "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)\n"
                "function global:wslc {\n"
                "    ConvertTo-Json -InputObject @($args) -Compress | Add-Content $env:TRACE -Encoding UTF8\n"
                "    if ($args[0] -eq 'image') { $global:LASTEXITCODE = [int]$env:MISSING }\n"
                "    else { $global:LASTEXITCODE = [int]$env:STATUS }\n}\n"
                "function global:Get-CimInstance { return @{Name='Intel Iris Xe'} }\n"
                "try { & $env:LAUNCHER -Wsl " + arguments
                + ' } catch { Write-Error $_; exit 1 }\nexit $LASTEXITCODE\n', encoding='utf-8-sig')
            result = subprocess.run([POWERSHELL, '-NoProfile', '-File', str(script)],
                capture_output=True, env={**os.environ, 'TRACE': str(trace),
                    'MISSING': str(missing), 'STATUS': str(status),
                    'LAUNCHER': str(ROOT / 'run_tagger.ps1'), 'IMAGES': folder})
            calls = [json.loads(line) for line in trace.read_text(encoding='utf-8-sig').splitlines()] if trace.exists() else []
            return result, calls

    def test_server_and_gpu_options(self):
        result, calls = self.launch('-s -ep intel -m ultra -gi 2 -u 5100 -pa 0.0.0.0')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = calls[-1]
        self.assertEqual(args[args.index('--gpus') + 1], 'all')
        self.assertEqual(args[args.index('--publish') + 1], '0.0.0.0:5100:5100')
        self.assertEqual(args[args.index('--gpu-index') + 1], '2')
        self.assertNotIn('--directml-device-index', args)
        self.assertIn('--server', args)

    def test_standalone_arguments_and_unicode_mount(self):
        result, calls = self.launch('-p $env:IMAGES -b 2 -w 0 -q 0 -o -z -r -np fp32 -ns 0 -ra @("--force")')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = calls[-1]
        for flag in ['/images', '--no-tag', '--organize', '--force', '--no-report', '--recursive']:
            self.assertIn(flag, args)
        self.assertNotIn('--server', args)
        for key, value in {'--batch-size': '2', '--io-workers': '0', '--thresh': '0', '--ncnn-part-size-mib': '0'}.items():
            self.assertEqual(args[args.index(key) + 1], value)
        self.assertTrue(any('日本語 ' in arg for arg in args))
        result, calls = self.launch("-p (Join-Path $env:IMAGES 'invoke.ps1')")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('/images/invoke.ps1', calls[-1])

    def test_client_and_upload_mode(self):
        result, calls = self.launch('-c -j 192.0.2.1 -p $env:IMAGES -um o -ep cuda')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = calls[-1]
        self.assertIn('--client', args)
        self.assertEqual(args[args.index('--host') + 1], '192.0.2.1')
        self.assertEqual(args[args.index('--client-upload-mode') + 1], 'o')
        self.assertNotIn('--gpus', args)

    def test_initial_image_build_and_failure_propagation(self):
        result, calls = self.launch('-pr -ep intel', missing=1)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[0] for call in calls], ['image', 'build', 'run'])
        result, calls = self.launch('-pr', missing=1, status=7)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual([call[0] for call in calls], ['image', 'build'])
        result, calls = self.launch('-pr', status=7)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual([call[0] for call in calls], ['image', 'run'])

    def test_explicit_build(self):
        result, calls = self.launch('-ac Build -bi vendor/base:version')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(calls), 1)
        self.assertIn('BASE_IMAGE=vendor/base:version', calls[0])

    def test_setup_only_and_login(self):
        result, calls = self.launch()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--gen-config', calls[-1])
        result, calls = self.launch('-lo -vol custom-volume -dp $env:IMAGES -gr $env:IMAGES')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--login', calls[-1])
        self.assertIn('--interactive', calls[-1])
        self.assertIn('type=volume,source=custom-volume,target=/workspace', calls[-1])

    def test_help_does_not_start_container(self):
        result, calls = self.launch('-h')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls, [])
        self.assertIn('-WslAction (-ac)', result.stdout.decode('utf-8'))

    def test_invalid_modes_and_directml(self):
        for options in ['-s -c', '-md Probe -s', '-ep directml', '-di 0', '-c']:
            result, calls = self.launch(options)
            self.assertNotEqual(result.returncode, 0, options)
            self.assertFalse(any(call[0] == 'run' for call in calls))

    def test_gpu_auto_selection(self):
        result, calls = self.launch('-s -g')
        self.assertEqual(result.returncode, 0, result.stderr)
        args = calls[-1]
        self.assertEqual(args[args.index('--provider') + 1], 'intel')

    def test_utf8_bom(self):
        for name in ['run_tagger.ps1', 'setup_tagger_wsl.ps1']:
            self.assertTrue((ROOT / name).read_bytes().startswith(b'\xef\xbb\xbf'))


if __name__ == '__main__':
    unittest.main()
