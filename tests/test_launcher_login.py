"""Login environment discovery without credentials, package downloads or GPUs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]

@unittest.skipUnless(os.name != 'nt' and shutil.which('bash'), 'POSIX Bash required')
class LoginDiscoveryTests(unittest.TestCase):
    def run_login(self, environments, login_status=0):
        with tempfile.TemporaryDirectory() as work:
            root = Path(work)
            shutil.copyfile(ROOT / 'run_tagger.sh', root / 'run_tagger.sh')
            trace = root / 'trace'
            for name, state in environments:
                python = root / name / 'bin/python'
                python.parent.mkdir(parents=True)
                python.write_text('#!/bin/bash\n'
                    'echo "' + name + ' $*" >> "$TRACE"\n'
                    + ('exit 9\n' if state == 'broken' else
                    'if [[ "$1" == -c ]]; then exit 0; fi\n'
                    + ('if [[ "$*" == *"--help"* ]]; then exit 3; fi\n' if state == 'missing' else '')
                    + 'if [[ "$*" == *"auth login"* ]]; then exit "$LOGIN_STATUS"; fi\nexit 0\n'))
                python.chmod(0o755)
            result = subprocess.run(['bash', str(root / 'run_tagger.sh'), '--login'],
                text=True, capture_output=True, env={**os.environ, 'TRACE': str(trace),
                'LOGIN_STATUS': str(login_status)})
            return result, trace.read_text()

    def test_skips_broken_and_missing_cli_before_healthy_rocm(self):
        result, trace = self.run_login([('venv_amd', 'broken'),
            ('venv_ncnn', 'missing'), ('venv_rocm', 'ready')])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('venv_rocm -m huggingface_hub.cli.hf auth login', trace)
        self.assertNotIn('pip install', trace)

    def test_discovers_custom_and_dot_venv_without_hf_script(self):
        for name in ['venv_custom', '.venv', 'venv_migraphx', 'venv_directml']:
            with self.subTest(name=name):
                result, trace = self.run_login([(name, 'ready')])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(name + ' -m huggingface_hub.cli.hf auth login', trace)

    def test_login_failure_is_returned(self):
        result, trace = self.run_login([('venv_intel', 'ready')], 7)
        self.assertEqual(result.returncode, 7)
        self.assertNotIn('ログインが完了しました', result.stdout)

@unittest.skipUnless(shutil.which('powershell') or shutil.which('pwsh'), 'PowerShell required')
class PowerShellLoginDiscoveryTests(unittest.TestCase):
    def test_skips_broken_environment_and_preserves_login_exit(self):
        source = (ROOT / 'run_tagger.ps1').read_text()
        start = source.index("if ($Login -or")
        end = source.index('\n}\n', start) + 2
        block = source[start:end]
        for status in [0, 7]:
            script = r'''$Login=$true
$ScriptDir='unused'
$IsWindowsOS=$true
function Get-ChildItem {
    [PSCustomObject]@{Name='venv_broken';FullName='venv_broken'}
    [PSCustomObject]@{Name='venv_rocm';FullName='venv_rocm'}
}
function Join-Path { param($Path,$ChildPath) if ($Path -eq 'venv_broken') {'BrokenPython'} else {'ReadyPython'} }
function Test-Path { return $true }
function BrokenPython { $global:LASTEXITCODE=9 }
function ReadyPython {
    Write-Output ('TRACE: '+($args -join ' '))
    if ($args -contains 'login') { $global:LASTEXITCODE=LOGIN_STATUS }
    else { $global:LASTEXITCODE=0 }
}
'''.replace('LOGIN_STATUS', str(status)) + block
            with tempfile.TemporaryDirectory() as work:
                path = Path(work) / 'probe.ps1'
                path.write_text(script, encoding='utf-8-sig')
                result = subprocess.run([shutil.which('powershell') or shutil.which('pwsh'),
                    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', str(path)],
                    capture_output=True)
            self.assertEqual(result.returncode, status, result.stdout + result.stderr)
            self.assertIn(b'TRACE: -m huggingface_hub.cli.hf auth login', result.stdout)

if __name__ == '__main__':
    unittest.main()
