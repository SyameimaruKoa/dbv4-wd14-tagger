"""Opt-in native tests of byte preservation and streamed graph boundaries."""

import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from ncnn_backend import StreamingNcnnRuntimeModel
from partition_ncnn_model import partition_model


@unittest.skipUnless(os.environ.get('DBV4_NCNN_PARTITION_TEST') == '1',
                     'requires native ncnn and a Vulkan GPU')
class PartitionTests(unittest.TestCase):
    def write_model(self, directory):
        prefix = Path(directory) / 'model'
        prefix.with_suffix('.ncnn.param').write_text('''7767517
8 9
Input input 0 1 in0
Convolution conv1 1 1 in0 a 0=1 1=1 5=1 6=3
Split split 1 2 a left right
Sigmoid sigmoid1 1 1 left b
Sigmoid sigmoid2 1 1 right c
BinaryOp add 2 1 b c d 0=0
Convolution conv2 1 1 d e 0=1 1=1 5=1 6=1
Sigmoid sigmoid3 1 1 e out0
''')
        weights = struct.pack('<I4fI2f', 0, .25, .5, -.25, .125, 0, 2., .1)
        prefix.with_suffix('.ncnn.bin').write_bytes(weights)
        return prefix, weights

    def test_weights_preserved_branch_retained_and_results_match_analytic(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, weights = self.write_model(directory)
            cache, parts = partition_model(prefix, .00001)
            self.assertEqual(b''.join((cache / p['bin']).read_bytes() for p in parts), weights)
            self.assertEqual(parts[1]['input'], 'a')
            # The residual fan-out remains wholly inside one section.
            text = (cache / parts[1]['param']).read_text()
            self.assertIn('Split split', text)
            self.assertIn('BinaryOp add', text)
            self.assertEqual(partition_model(prefix, .00001), (cache, parts))
            x = np.linspace(-1, 1, 48, dtype=np.float32).reshape(3, 4, 4)
            metadata = SimpleNamespace(label_count=16, profile_name='synthetic')
            runtime = StreamingNcnnRuntimeModel(metadata, lambda _: x, prefix,
                                               part_size_mib=.00001)
            try:
                actual = runtime.predict_preprocessed(x[None])[0]
            finally:
                runtime.close()
            intermediate = 1 / (1 + np.exp(-(.25*x[0] + .5*x[1] - .25*x[2] + .125)))
            expected = 1 / (1 + np.exp(-(4*intermediate + .1)))
            np.testing.assert_allclose(actual, expected.reshape(-1), atol=1e-6, rtol=1e-6)

    def test_non_fp32_weights_rejected_without_publishing_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, weights = self.write_model(directory)
            prefix.with_suffix('.ncnn.bin').write_bytes(struct.pack('<I', 0x01306B47) + weights[4:])
            with self.assertRaisesRegex(ValueError, 'Only FP32 weights'):
                partition_model(prefix, 128)
            self.assertEqual(list(Path(directory).glob('model.parts-*')), [])
