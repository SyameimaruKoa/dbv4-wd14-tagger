"""Opt-in real ultra/Vulkan tests; requires ncnn, HF access and ExifTool.

DBV4_NCNN_INTEGRATION=1 DBV4_NCNN_REFERENCE=benchmarks/ncnn_ultra/cpu-b1.json \
    venv_ncnn/bin/python -m unittest discover -s tests -p test_ncnn_integration.py -v
"""

import copy
import json
import os
from pathlib import Path
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

import embed_tags_universal as app
from benchmark_ncnn import benchmark_image, compare


@unittest.skipUnless(os.environ.get('DBV4_NCNN_INTEGRATION') == '1',
                     'set DBV4_NCNN_INTEGRATION=1 on a Vulkan GPU host')
class NcnnIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = app.load_runtime_model(True, 'ultra', provider='ncnn')
        cls.image = benchmark_image()
        cls.probabilities = cls.runtime.predict_images([cls.image])[0]

    def test_cpu_probability_rating_and_threshold_parity(self):
        reference_path = os.environ.get('DBV4_NCNN_REFERENCE')
        if not reference_path:
            self.skipTest('set DBV4_NCNN_REFERENCE to the CPU benchmark JSON')
        reference = json.loads(Path(reference_path).read_text())
        self.assertEqual(self.probabilities.shape, (12476,))
        np.testing.assert_allclose(self.probabilities, reference['probabilities'],
                                   atol=1e-4, rtol=1e-3)
        differences = compare(reference, self.probabilities, self.runtime.metadata)
        self.assertEqual(differences['selected_tag_symmetric_difference'], [])
        self.assertEqual(len(differences['rating_absolute_difference']), 4)
        self.assertLess(max(differences['rating_absolute_difference'].values()), 1e-4)

    @unittest.skipUnless(shutil.which('exiftool'), 'ExifTool is required for XMP validation')
    def test_standalone_and_both_client_modes_write_same_xmp(self):
        class Handler(app.TagServerHandler):
            pass

        Handler.runtime = self.runtime
        server = app.ParallelTagServer(('127.0.0.1', 0), Handler, 1)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(
                app, 'APP_CONFIG', copy.deepcopy(app.DEFAULT_CONFIG)
            ):
                written = []
                for mode in ('standalone', 'original', 'preprocessed'):
                    path = Path(directory) / (mode + '.png')
                    self.image.save(path)
                    options = ['--model-profile', 'ultra', '--no-report', '--force',
                               '--batch-size', '4', '--io-workers', '0', str(path)]
                    if mode != 'standalone':
                        options += ['--mode', 'client', '--host', '127.0.0.1',
                                    '--port', str(server.server_port),
                                    '--client-upload-mode', mode]
                    args = app.create_parser().parse_args(options)
                    # Reuse the real GPU runtime to avoid reloading 2.58 GiB.
                    with patch.object(app, 'load_runtime_model', return_value=self.runtime):
                        app.process_images(args)
                    tags = app.et_wrapper.get_tags(str(path))
                    app.et_wrapper.stop()
                    self.assertTrue(tags)
                    self.assertTrue(set(self.runtime.metadata.decode_tags(
                        self.probabilities)).issubset(tags))
                    self.assertTrue(set(app.format_score_tags(self.runtime.metadata,
                                                             self.probabilities)).issubset(tags))
                    written.append(set(tags))
                self.assertEqual(written[0], written[1])
                self.assertEqual(written[0], written[2])
        finally:
            app.et_wrapper.stop()
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == '__main__':
    unittest.main()
