"""Exercise the Bash provider selection with failing runtime executables."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


@unittest.skipUnless(os.name != 'nt' and shutil.which('bash'), 'POSIX Bash is required')
class LauncherFallbackTests(unittest.TestCase):
    def select(self, vendor, succeeds):
        source = (Path(__file__).resolve().parents[1] / 'run_tagger.sh').read_text()
        functions = '\n'.join(re.search(
            rf'(?ms)^{name}\(\) \{{.*?^\}}', source
        ).group() for name in ('auto_backend_mode', 'auto_venv_name', 'select_auto_provider'))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            executable = '''#!/usr/bin/env bash
if [[ "$1" == -c ]]; then exit 0; fi
while [[ $# -gt 0 ]]; do
    if [[ "$1" == --provider ]]; then provider="$2"; break; fi
    shift
done
printf '%s\\n' "$provider" >> "$TRACE"
[[ "$provider" == "$SUCCEEDS" ]]
'''
            for venv in ('venv_gpu', 'venv_intel', 'venv_migraphx', 'venv_rocm',
                         'venv_ncnn', 'venv_webgpu', 'venv_std'):
                python = root / venv / 'bin/python'
                python.parent.mkdir(parents=True)
                python.write_text(executable)
                python.chmod(0o755)
            script = functions + '''
setup_env() { return 0; }
SCRIPT_DIR="$WORK"
PYTHON_SCRIPT=embed_tags_universal.py
PY_ARGS=(--model-profile ultra)
IS_CLIENT=0
select_auto_provider "$VENDOR"
printf 'selected=%s\\n' "$PROVIDER"
'''
            result = subprocess.run(['bash', '-c', script], text=True, capture_output=True,
                                    env={**os.environ, 'WORK': directory, 'VENDOR': vendor,
                                         'TRACE': str(root / 'trace'), 'SUCCEEDS': succeeds})
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('selected=' + succeeds, result.stdout)
            return (root / 'trace').read_text().splitlines()

    def test_failing_gpu_initializations_follow_vendor_priorities(self):
        for vendor, expected in (
            ('nvidia', ['tensorrt', 'cuda', 'ncnn', 'webgpu', 'cpu']),
            ('intel', ['intel', 'ncnn', 'webgpu', 'cpu']),
            ('amd', ['migraphx', 'rocm', 'ncnn', 'webgpu', 'cpu']),
            ('amd_unsupported', ['ncnn', 'webgpu', 'cpu']),
            ('ps4', ['ncnn', 'webgpu', 'cpu']),
            ('switch', ['ncnn', 'webgpu', 'cpu']),
        ):
            with self.subTest(vendor=vendor):
                self.assertEqual(self.select(vendor, 'cpu'), expected)

    def test_healthy_dedicated_runtime_remains_first_choice(self):
        for vendor, provider in (('nvidia', 'tensorrt'), ('intel', 'intel'), ('amd', 'migraphx')):
            with self.subTest(vendor=vendor):
                self.assertEqual(self.select(vendor, provider), [provider])

    def test_ncnn_initialization_failure_reaches_webgpu_before_cpu(self):
        self.assertEqual(self.select('amd_unsupported', 'webgpu'), ['ncnn', 'webgpu'])


if __name__ == '__main__':
    unittest.main()
