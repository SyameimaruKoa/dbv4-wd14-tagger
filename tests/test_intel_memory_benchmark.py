"""Exercise failed and timed-out measurements with actual child processes."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import benchmark_intel_memory as benchmark


class IntelMemoryBenchmarkTests(unittest.TestCase):
    def measure_stub(self, code, timeout):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            stub = root / "child.py"
            stub.write_text(code)
            args = SimpleNamespace(model="unused.onnx", device="GPU.0", iterations=1,
                                   baseline_batch_size=4, timeout=timeout)
            output = root / "measurement"
            with patch.object(benchmark, "__file__", str(stub)):
                result = benchmark.measure(args, "baseline", output)
            samples = json.loads((output / "samples.json").read_text())
            return result, samples, (output / "run.log").read_text()

    def test_failed_child_preserves_memory_samples_and_diagnostic(self):
        result, samples, log = self.measure_stub(
            "import sys; print('GPU initialization failed', flush=True); sys.exit(7)", 5)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["exit_code"], 7)
        self.assertTrue(samples)
        self.assertIn("GPU initialization failed", log)

    def test_timeout_terminates_child_and_preserves_partial_measurement(self):
        result, samples, _ = self.measure_stub("import time; time.sleep(30)", 0.3)
        self.assertEqual(result["status"], "timeout")
        self.assertNotEqual(result["exit_code"], 0)
        self.assertTrue(samples)
        self.assertIn("min_available_mib", result)


if __name__ == "__main__":
    unittest.main()
