import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from convert_ncnn_model import convert
from ncnn_backend import NcnnRuntimeModel, ensure_ncnn_model, vulkan_device


class FakeExtractor:
    def __init__(self, output):
        self.output = output
        self.input_tensor = None

    def input(self, name, mat):
        self.input_tensor = (name, np.asarray(mat))
        return 0

    def extract(self, name):
        return 0, self.output


class FakeNet:
    def __init__(self):
        self.opt = SimpleNamespace()
        self.extractors = []
        self.device = None

    def set_vulkan_device(self, index):
        self.device = index

    def load_param(self, path):
        return 0

    def load_model(self, path):
        return 0

    def input_names(self):
        return ["in0"]

    def output_names(self):
        return ["out0"]

    def create_extractor(self):
        result = FakeExtractor(np.array([0.1, 0.4, 0.8, 0.9], np.float32))
        self.extractors.append(result)
        return result


class NcnnBackendTests(unittest.TestCase):
    def test_runtime_returns_dbv4_probabilities_for_each_image(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "model"
            prefix.with_suffix(".ncnn.param").write_text("param")
            prefix.with_suffix(".ncnn.bin").write_bytes(b"bin")
            net = FakeNet()
            fake_ncnn = SimpleNamespace(Net=lambda: net, Mat=lambda value: value,
                                        get_gpu_count=lambda: 1,
                                        get_gpu_info=lambda index: SimpleNamespace(
                                            device_name=lambda: "test GPU", type=lambda: 2))
            metadata = SimpleNamespace(label_count=4, profile_name="ultra")
            preprocessor = lambda image: np.ones((3, 4, 4), np.float32)
            with patch.dict(sys.modules, {"ncnn": fake_ncnn}):
                runtime = NcnnRuntimeModel(metadata, preprocessor, prefix,
                                           precision="fp16-packed")
                result = runtime.predict_images([Image.new("RGB", (4, 4))] * 2)
            self.assertEqual(runtime.batch_limit, 1)
            self.assertEqual(len(result), 2)
            np.testing.assert_allclose(result[0], [0.1, 0.4, 0.8, 0.9])
            self.assertEqual(net.extractors[0].input_tensor[1].shape, (3, 4, 4))
            self.assertTrue(net.opt.use_vulkan_compute)
            self.assertTrue(net.opt.use_fp16_storage)
            self.assertTrue(net.opt.use_fp16_packed)
            self.assertFalse(net.opt.use_fp16_arithmetic)

    def test_software_vulkan_device_is_rejected(self):
        ncnn = SimpleNamespace(get_gpu_count=lambda: 1,
                               get_gpu_info=lambda index: SimpleNamespace(
                                   device_name=lambda: "llvmpipe", type=lambda: 3))
        with self.assertRaisesRegex(RuntimeError, "ソフトウェアデバイス"):
            vulkan_device(ncnn, 0)

    def test_cached_converter_files_do_not_require_conversion_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            dest = Path(directory)
            (dest / "model.ncnn.param").write_text("param")
            (dest / "model.ncnn.bin").write_bytes(b"bin")
            (dest / "model.ncnn.json").write_text(json.dumps({
                "source_repo": "owner/model", "input_size": 512,
                "label_count": 4, "fp16": False,
            }))
            ensure_ncnn_model(dest / "model", "owner/model", 512, 4)
            self.assertEqual(convert("owner/model", dest, 512, 4),
                             (dest / "model.ncnn.param", dest / "model.ncnn.bin"))


if __name__ == "__main__":
    unittest.main()
