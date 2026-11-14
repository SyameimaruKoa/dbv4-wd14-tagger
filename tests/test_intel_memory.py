import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

import embed_tags_universal as app


class IntelMemoryTests(unittest.TestCase):
    def runtime(self, profile="ultra", provider="OpenVINOExecutionProvider", batch=None):
        session = MagicMock()
        session.get_inputs.return_value = [SimpleNamespace(name="input", shape=[batch, 3, 8, 8])]
        session.get_outputs.return_value = [SimpleNamespace(name="probabilities", shape=[batch, 2])]
        session.get_providers.return_value = [provider, "CPUExecutionProvider"]
        session.run.return_value = [np.array([[0.25, 0.75]], dtype=np.float32)]
        metadata = SimpleNamespace(profile_name=profile, label_count=2)
        return app.RuntimeModel(metadata, lambda image: np.zeros((3, 8, 8), dtype=np.float32), session)

    def test_ultra_limit_is_advertised_to_server_and_rejects_direct_batches(self):
        runtime = self.runtime()
        self.assertEqual(runtime.batch_limit, 1)
        handler = object.__new__(app.TagServerHandler)
        handler.runtime = runtime
        with patch.dict(app.APP_CONFIG, {"server_max_batch_images": 8}):
            self.assertEqual(handler._batch_limit(), 1)
        with self.assertRaisesRegex(ValueError, "1枚"):
            runtime.predict_preprocessed(np.zeros((4, 3, 8, 8), dtype=np.float32))
        runtime.session.run.assert_not_called()

    def test_other_providers_profiles_and_fixed_shapes_keep_batch_limits(self):
        self.assertIsNone(self.runtime(provider="CUDAExecutionProvider").batch_limit)
        self.assertIsNone(self.runtime(profile="balanced").batch_limit)
        self.assertEqual(self.runtime(batch=2).batch_limit, 2)

    def test_concurrent_requests_cannot_enter_session_together(self):
        runtime = self.runtime()
        entered = threading.Event()
        second_started = threading.Event()
        release = threading.Event()
        calls = []

        def run(*args):
            calls.append(1)
            entered.set()
            self.assertTrue(release.wait(5))
            return [np.array([[0.25, 0.75]], dtype=np.float32)]

        def second():
            second_started.set()
            return runtime.predict_images([None])

        runtime.session.run.side_effect = run
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(runtime.predict_images, [None])
            try:
                self.assertTrue(entered.wait(5))
                # The first request holds the same lock used by both public paths.
                self.assertFalse(runtime._inference_lock.acquire(blocking=False))
                next_request = pool.submit(second)
                self.assertTrue(second_started.wait(5))
                self.assertEqual(len(calls), 1)
            finally:
                release.set()
            np.testing.assert_array_equal(first.result()[0], next_request.result()[0])
        self.assertEqual(len(calls), 2)


if __name__ == "__main__":
    unittest.main()
