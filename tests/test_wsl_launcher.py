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
                "[Console]::OutputEncoding = [Text.UTF8Encoding]::new($false)\n"
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
                output = result.stdout.decode('utf-8')
                for heading in ['DBV4 Tagger Universal (日本語ヘルプ)', '使い方:',
                                '処理時に必要な入力:', '値を指定するオプション（<>内の値が必要）:',
                                '値を指定しないスイッチ:', '実行例:']:
                    self.assertIn(heading, output)
                self.assertNotIn('SYNTAX', output)
                provider_line = next(line for line in output.splitlines() if '-Provider (-ep)' in line)
                self.assertIn('cpu / cuda / tensorrt', provider_line)
                self.assertIn('（既定 cpu）', provider_line)

    def test_powershell_source_has_utf8_bom(self):
        self.assertTrue((ROOT / 'run_tagger_wsl.ps1').read_bytes().startswith(b'\xef\xbb\xbf'))

    def test_common_short_options_forward_values_and_flags(self):
        result, args = self.launch("-md Standalone -p $env:WSL_TEST_IMAGES -ep intel -m ultra -gi 1 -b 2 -w 0 -q 0 -e owner/model -l /models/model.onnx -y /models/tags.csv -np fp32 -ns 0 -wi 1 -tv intel -od GPU.1 -td /runtime -v 4 -d 0.5 -wmm R-17_0 -o -t -x -z -r -f -a -i")
        self.assertEqual(result.returncode, 0, result.stderr)
        for option, value in {'--batch-size': '2', '--io-workers': '0', '--thresh': '0',
                              '--model-repo': 'owner/model', '--model-file': '/models/model.onnx',
                              '--tags-file': '/models/tags.csv', '--ncnn-part-size-mib': '0',
                              '--openvino-device': 'GPU.1', '--wallgen-move-min-rating': 'R-17_0'}.items():
            self.assertEqual(args[args.index(option) + 1], value)
        for flag in ['--organize', '--tag', '--wallgen-unsorted', '--no-report',
                     '--recursive', '--force', '--record-ratio', '--ignore-sensitive']:
            self.assertIn(flag, args)

    def test_mode_and_host_aliases_match_native_launcher(self):
        result, args = self.launch('-c -j 192.0.2.1 -u 5100 -p $env:WSL_TEST_IMAGES -um o -n -k')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--client', args)
        self.assertEqual(args[args.index('--host') + 1], '192.0.2.1')
        self.assertEqual(args[args.index('--port') + 1], '5100')
        self.assertEqual(args[args.index('--client-upload-mode') + 1], 'o')
        self.assertIn('--no-recursive', args)
        self.assertIn('--no-record-ratio', args)
        result, args = self.launch('-c -HostIP 192.0.2.1 -p $env:WSL_TEST_IMAGES')
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_container_short_options_and_login(self):
        result, args = self.launch('-lo -im dbv4:test -vol test-volume -cn test-container -dp $env:WSL_TEST_IMAGES -gr $env:WSL_TEST_IMAGES -it -ta @("--debug")')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--login', args)
        self.assertIn('--interactive', args)
        self.assertIn('dbv4:test', args)
        self.assertIn('type=volume,source=test-volume,target=/workspace', args)
        self.assertEqual(args[args.index('--name') + 1], 'test-container')
        result, args = self.launch('-ac Build -bi vendor/base:version')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('BASE_IMAGE=vendor/base:version', args)

    def test_conflicting_modes_and_directml_stop_before_container(self):
        for options in ['-s -c', '-md Probe -s', '-ep directml', '-di 0', '-wg -ep intel']:
            with self.subTest(options=options):
                result, args = self.launch(options)
                self.assertNotEqual(result.returncode, 0)
                self.assertIsNone(args)

    def test_short_gpu_switch_and_server(self):
        result, args = self.launch('-s -g -ep intel -pa 0.0.0.0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--server', args)
        self.assertIn('--gpus', args)
        result, args = self.launch('-wg -md Probe')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(args[args.index('--provider') + 1], 'webgpu')

    def test_path_without_mode_selects_standalone(self):
        result, args = self.launch('-p $env:WSL_TEST_IMAGES -ep cpu -b 1')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--path', args)
        self.assertNotIn('--server', args)
        self.assertNotIn('--publish', args)

    def test_every_user_parameter_has_alias_and_common_aliases_match(self):
        wsl_path = str(ROOT / 'run_tagger_wsl.ps1').replace("'", "''")
        native_path = str(ROOT / 'run_tagger.ps1').replace("'", "''")
        command = f"$wsl = Get-Command '{wsl_path}'; $native = Get-Command '{native_path}'; " + r'''
            $common = @('Verbose','Debug','ErrorAction','WarningAction','InformationAction',
                'ProgressAction','ErrorVariable','WarningVariable','InformationVariable',
                'OutVariable','OutBuffer','PipelineVariable')
            foreach ($parameter in $wsl.Parameters.Values) {
                if ($parameter.Name -notin $common -and $parameter.Aliases.Count -eq 0) {
                    throw "Missing alias: $($parameter.Name)"
                }
            }
            foreach ($parameter in $native.Parameters.Values) {
                if ($parameter.Name -in $common) { continue }
                $name = $parameter.Name
                if ($name -eq 'HostIP') { $name = 'HostName' }
                if (-not $wsl.Parameters.ContainsKey($name)) { throw "Missing parameter: $name" }
                foreach ($alias in $parameter.Aliases) {
                    if (-not $alias.StartsWith('-') -and $alias -notin $wsl.Parameters[$name].Aliases) {
                        throw "Different alias: $name $alias"
                    }
                }
            }
        '''
        result = subprocess.run([POWERSHELL, '-NoProfile', '-Command', command], capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == '__main__':
    unittest.main()
