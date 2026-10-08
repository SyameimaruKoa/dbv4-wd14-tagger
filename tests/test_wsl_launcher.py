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
    def launch(self, arguments='', missing=0, status=0, existing=None):
        with tempfile.TemporaryDirectory(prefix='WSL 日本語 ') as folder:
            trace = Path(folder) / 'trace.json'
            script = Path(folder) / 'invoke.ps1'
            state = Path(folder) / 'existing.json'
            if existing is not None:
                state.write_text(json.dumps([existing]), encoding='utf-8-sig')
            script.write_text(
                "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)\n"
                "function global:wslc {\n"
                "    ConvertTo-Json -InputObject @($args) -Compress | Add-Content $env:TRACE -Encoding UTF8\n"
                "    if ($args[0] -eq 'image') { $global:LASTEXITCODE = [int]$env:MISSING }\n"
                "    elseif ($args[0] -eq 'container' -and $args[1] -eq 'inspect') {\n"
                "        if (Test-Path $env:EXISTING) { Get-Content $env:EXISTING -Raw; $global:LASTEXITCODE = 0 }\n"
                "        else { $global:LASTEXITCODE = 1 }\n"
                "    }\n"
                "    else { $global:LASTEXITCODE = [int]$env:STATUS }\n}\n"
                "function global:Get-CimInstance { return @{Name='Intel Iris Xe'} }\n"
                "try { & $env:LAUNCHER -Wsl " + arguments
                + ' } catch { Write-Error $_; exit 1 }\nexit $LASTEXITCODE\n', encoding='utf-8-sig')
            result = subprocess.run([POWERSHELL, '-NoProfile', '-File', str(script)],
                capture_output=True, env={**os.environ, 'TRACE': str(trace),
                    'MISSING': str(missing), 'STATUS': str(status),
                    'LAUNCHER': str(ROOT / 'run_tagger.ps1'), 'IMAGES': folder, 'EXISTING': str(state)})
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
        output = result.stdout.decode('utf-8')
        self.assertIn('実行モード: WSL / Linuxコンテナ (-Wsl)', output)
        self.assertIn('コンテナイメージ: dbv4-tagger-wsl:local', output)
        self.assertIn('サーバー公開設定: http://0.0.0.0:5100', output)
        self.assertIn('以降はコンテナ内のログです。', output)

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
        self.assertEqual([call[0] for call in calls], ['image', 'build', 'container', 'run'])
        result, calls = self.launch('-pr', missing=1, status=7)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual([call[0] for call in calls], ['image', 'build'])
        result, calls = self.launch('-pr', status=7)
        self.assertEqual(result.returncode, 7, result.stderr)
        self.assertEqual([call[0] for call in calls], ['image', 'container', 'run'])

    def existing_server(self, running=True):
        return {
            'Id': 'existing-id',
            'State': {'Running': running},
            'Config': {'Image': 'dbv4-tagger-wsl:local',
                'Entrypoint': ['/usr/local/bin/dbv4-container'],
                'Cmd': ['--server', '--provider', 'intel', '--gpu-index', '0',
                    '--ncnn-precision', 'fp32', '--model-profile', 'ultra', '--port', '5000']},
            'Mounts': [{'Destination': '/workspace', 'Type': 'volume',
                'Name': 'dbv4-tagger-wsl-workspace', 'ReadWrite': True}],
            'Ports': {'5000/tcp': [{'HostIp': '127.0.0.1', 'HostPort': '5000'}]},
        }

    def test_running_matching_server_is_reused(self):
        result, calls = self.launch('-s -ep intel -m ultra', existing=self.existing_server())
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual([call[0] for call in calls], ['image', 'container', 'logs'])
        self.assertIn('既に稼働中', result.stdout.decode('utf-8'))

    def test_stopped_managed_container_is_removed_without_deleting_data(self):
        result, calls = self.launch('-s -ep intel -m ultra', existing=self.existing_server(False))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(calls[-2], ['container', 'remove', 'existing-id'])
        self.assertEqual(calls[-1][0], 'run')

    def test_running_different_configuration_is_not_stopped(self):
        for difference in ['model', 'mount', 'port']:
            state = self.existing_server()
            if difference == 'model':
                state['Config']['Cmd'][-3] = 'balanced'
            elif difference == 'mount':
                state['Mounts'].append({'Destination': '/other', 'Type': 'bind'})
            else:
                state['Ports']['5000/tcp'][0]['HostIp'] = '0.0.0.0'
            result, calls = self.launch('-s -ep intel -m ultra', existing=state)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual([call[0] for call in calls], ['image', 'container'])

    def test_foreign_stopped_container_is_preserved(self):
        state = self.existing_server(False)
        state['Config']['Image'] = 'another-app:latest'
        result, calls = self.launch('-s -ep intel -m ultra', existing=state)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual([call[0] for call in calls], ['image', 'container'])

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
