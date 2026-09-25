import base64
import gzip
import io
import json
import socket
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from dbv4 import DBV4Preprocessor

from embed_tags_universal import (
    ClientBatchPipeline,
    ClientCompatibilityError,
    DEFAULT_CONFIG,
    ParallelTagServer,
    PartialBatchPredictions,
    RuntimeModel,
    TagServerHandler,
    align_client_model,
    client_predict_batch,
    client_predict_tensor_batch,
    compatible_preprocess_format,
    create_parser,
    migrate_legacy_config,
    preprocessor_probe_hash,
    server_http_error,
)


class BatchProtocolTests(unittest.TestCase):
    def test_codec_mismatch_only_disables_affected_format(self):
        client = {"jpg": "6.2", "webp": "1.6.0", "zlib": "1.3.1", "AVIF": "1.3.0"}
        server = {**client, "jpg": "8.0"}
        self.assertFalse(compatible_preprocess_format("a.jpg", client, server))
        self.assertTrue(compatible_preprocess_format("b.webp", client, server))
        self.assertTrue(compatible_preprocess_format("c.png", client, server))
        self.assertTrue(compatible_preprocess_format("d.bmp", client, server))

    def test_legacy_pixel_limit_is_migrated(self):
        config = {"server_max_image_pixels": 20000000, "model_profiles": {}}
        self.assertTrue(migrate_legacy_config(config))
        self.assertEqual(config["server_max_image_pixels"], 80000000)

    def test_transfer_mode_defaults_to_preprocessed_and_has_override(self):
        self.assertEqual(DEFAULT_CONFIG["client_upload_mode"], "preprocessed")
        args = create_parser().parse_args(["--client-upload-mode", "original"])
        self.assertEqual(args.client_upload_mode, "original")
        self.assertEqual(create_parser().parse_args(["-U", "p"]).client_upload_mode, "p")
        self.assertEqual(create_parser().parse_args(["-um", "o"]).client_upload_mode, "o")

    def test_server_http_error_includes_response_detail(self):
        error = urllib.error.HTTPError(
            "http://server/batch", 500, "Internal Server Error", {},
            io.BytesIO(json.dumps({"error": "CUDA out of memory"}).encode("utf-8")),
        )
        self.assertEqual(server_http_error(error), "HTTP 500: CUDA out of memory")

    def test_old_lossy_upload_setting_migrates_to_original(self):
        config = {"client_upload_mode": "optimized", "model_profiles": {}}
        self.assertTrue(migrate_legacy_config(config))
        self.assertEqual(config["client_upload_mode"], "original")

    def test_preprocessed_tensor_matches_nhwc_model_input(self):
        preprocessor = DBV4Preprocessor({"test": [{"type": "Resize", "size": [8, 8]}]})
        metadata = SimpleNamespace(label_count=2)

        class Session:
            inputs = []

            def get_inputs(self):
                return [SimpleNamespace(name="input", shape=[None, 8, 8, 3])]

            def get_outputs(self):
                return [SimpleNamespace(name="probabilities", shape=[None, 2])]

            def run(self, names, feeds):
                self.inputs.append(feeds["input"].copy())
                return [np.asarray([[0.25, 0.75]], dtype=np.float32)]

        session = Session()
        runtime = RuntimeModel(metadata, preprocessor, session)
        image = Image.new("RGB", (12, 10), (20, 30, 40))
        runtime.predict_images([image])
        runtime.predict_preprocessed(np.stack([preprocessor(image)], axis=0))
        np.testing.assert_array_equal(session.inputs[0], session.inputs[1])

    def test_client_preprocessed_mode_matches_original_inference(self):
        preprocess_config = {"test": [
            {"type": "PadToSize", "size": [64, 64], "background_color": "white"},
            {"type": "Resize", "size": 32, "interpolation": "bicubic"},
            {"type": "CenterCrop", "size": [32, 32]},
            {"type": "Normalize", "mean": [0.5, 0.5, 0.5], "std": [0.25, 0.25, 0.25]},
        ]}
        metadata = SimpleNamespace(
            repo_id="test/model", metadata_version="v1", label_count=2,
            preprocess_config=preprocess_config,
            summary=lambda: {"protocol": 1, "model_id": "test/model", "metadata_version": "v1", "output_size": 2},
        )

        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="input", shape=[None, 3, 32, 32])]

            def get_outputs(self):
                return [SimpleNamespace(name="probabilities", shape=[None, 2])]

            def run(self, names, feeds):
                values = feeds["input"]
                return [np.stack((values.mean(axis=(1, 2, 3)), values.std(axis=(1, 2, 3))), axis=1)]

        preprocessor = DBV4Preprocessor.from_metadata(metadata)
        self.assertEqual(preprocessor_probe_hash(preprocessor), preprocessor_probe_hash(DBV4Preprocessor.from_metadata(metadata)))
        changed = DBV4Preprocessor({"test": [{"type": "Resize", "size": 32}]})
        self.assertNotEqual(preprocessor_probe_hash(preprocessor), preprocessor_probe_hash(changed))
        runtime = RuntimeModel(metadata, preprocessor, Session())
        with tempfile.TemporaryDirectory() as directory, patch.object(TagServerHandler, "runtime", runtime):
            path = Path(directory) / "image.png"
            transparent_path = Path(directory) / "transparent.png"
            pixels = np.random.default_rng(0).integers(0, 256, (40, 60, 3), dtype=np.uint8)
            Image.fromarray(pixels, "RGB").save(path)
            rgba = np.dstack((pixels, np.full((40, 60), 120, dtype=np.uint8)))
            Image.fromarray(rgba, "RGBA").save(transparent_path)
            server = ParallelTagServer(("127.0.0.1", 0), TagServerHandler, 2)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}"
                with urllib.request.urlopen(f"{url}/metadata") as response:
                    server_info = json.load(response)
                self.assertTrue(server_info["tensor_batch_supported"])
                self.assertEqual(server_info["tensor_preprocess_hash"], preprocessor_probe_hash(preprocessor))
                self.assertIn("Pillow", server_info["tensor_preprocess_environment"])
                paths = [str(path), str(transparent_path)]
                original = client_predict_batch(url, paths, metadata, 5)
                prepared = client_predict_tensor_batch(url, paths, metadata, preprocessor, 5)
                np.testing.assert_array_equal(prepared, original)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_server_receives_next_request_while_inference_runs(self):
        first_inference = threading.Event()
        second_received = threading.Event()
        second_inference = threading.Event()
        release_inference = threading.Event()

        class Runtime:
            batch_limit = 4
            metadata = SimpleNamespace(summary=lambda: {
                "protocol": 1, "model_id": "test/model", "metadata_version": "v1", "output_size": 2,
            })
            calls = 0

            def predict_images(self, images):
                self.calls += 1
                if self.calls == 1:
                    first_inference.set()
                    if not release_inference.wait(3):
                        raise TimeoutError("first inference was not released")
                else:
                    second_inference.set()
                return [np.asarray([1, 0], dtype=np.float32) for _ in images]

        class ObservedHandler(TagServerHandler):
            runtime = Runtime()
            reads = 0

            def _read_request_body(self, length, client_ip, image_name):
                body = super()._read_request_body(length, client_ip, image_name)
                type(self).reads += 1
                if type(self).reads == 2:
                    second_received.set()
                return body

        image = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        server = ParallelTagServer(("127.0.0.1", 0), ObservedHandler, 2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def send():
            with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/", data=image.getvalue(), timeout=5) as response:
                return response.status

        try:
            with ThreadPoolExecutor(max_workers=2) as requests:
                first = requests.submit(send)
                self.assertTrue(first_inference.wait(3))
                second = requests.submit(send)
                self.assertTrue(second_received.wait(3))
                self.assertTrue(second_inference.wait(3))
                release_inference.set()
                self.assertEqual((first.result(), second.result()), (200, 200))
        finally:
            release_inference.set()
            server.shutdown()
            server.server_close()
            thread.join()
        self.assertEqual(ObservedHandler.runtime.calls, 2)

    def test_metadata_responds_while_inference_slot_is_busy(self):
        entered = threading.Event()
        release = threading.Event()

        class Runtime:
            batch_limit = 2
            metadata = SimpleNamespace(summary=lambda: {
                "protocol": 1, "model_id": "test/model", "metadata_version": "v1", "output_size": 2,
            })

            def predict_images(self, images):
                entered.set()
                release.wait(5)
                return [np.asarray([1, 0], dtype=np.float32)]

        class Handler(TagServerHandler):
            runtime = Runtime()

        image = io.BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        server = ParallelTagServer(("127.0.0.1", 0), Handler, 1)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with ThreadPoolExecutor(max_workers=1) as requests:
                pending = requests.submit(
                    lambda: urllib.request.urlopen(
                        f"http://127.0.0.1:{server.server_port}/", data=image.getvalue(), timeout=5
                    ).close()
                )
                self.assertTrue(entered.wait(3))
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/metadata", timeout=2
                ) as response:
                    self.assertEqual(json.load(response)["model_id"], "test/model")
                release.set()
                pending.result(timeout=3)
        finally:
            release.set()
            server.shutdown()
            server.server_close()
            thread.join()

    def test_server_rejects_oversized_body_before_reading(self):
        handler = TagServerHandler.__new__(TagServerHandler)
        handler.server = SimpleNamespace(_worker_slots=threading.BoundedSemaphore(2))
        handler.runtime = SimpleNamespace(batch_limit=4)
        handler.path = "/batch"
        handler.client_address = ("127.0.0.1", 1000)
        handler.headers = {"Content-Length": str(129 * 1024 * 1024)}
        handler.rfile = io.BytesIO()
        responses = []
        handler._send_json_response = lambda status, body: responses.append(status) or True
        handler.do_POST()
        self.assertEqual(responses, [413])

    def test_server_rejects_oversized_decoded_image(self):
        handler = TagServerHandler.__new__(TagServerHandler)
        image = io.BytesIO()
        Image.new("RGB", (100, 100)).save(image, format="PNG")
        with patch.dict("embed_tags_universal.APP_CONFIG", {"server_max_image_pixels": 1000}):
            with self.assertRaisesRegex(ValueError, "画素数が上限"):
                handler._decode_image(image.getvalue())

    def test_client_rejects_oversized_batch_before_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.png"
            with path.open("wb") as image:
                image.truncate(100 * 1024 * 1024)
            with self.assertRaisesRegex(ValueError, "送信サイズ上限"):
                client_predict_batch("http://server:5000", [str(path)], None, 5)

    def test_server_logs_http_receive_rate(self):
        handler = TagServerHandler.__new__(TagServerHandler)
        handler.rfile = io.BytesIO(b"x" * (1024 * 1024))
        handler.headers = {}
        with patch("embed_tags_universal.time.perf_counter", side_effect=[10.0, 10.5]), patch(
            "builtins.print"
        ) as log:
            body = handler._read_request_body(1024 * 1024, "127.0.0.1", "image.png")
        self.assertEqual(len(body), 1024 * 1024)
        self.assertIn("HTTP受信: 1.00 MiB / 0.500s = 2.00 MiB/s", log.call_args.args[0])

    def test_pipeline_sends_next_batch_while_consuming_previous(self):
        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        consumed = []

        def request(items):
            if items == [1]:
                first_started.set()
                if not release_first.wait(3):
                    raise TimeoutError("first request was not released")
            else:
                second_started.set()
            return items

        def consume(items, future):
            if items == [1]:
                self.assertTrue(second_started.wait(2))
            consumed.append(future.result())

        pipeline = ClientBatchPipeline()
        try:
            pipeline.submit([1], request, consume)
            self.assertTrue(first_started.wait(3))
            submit_second = threading.Thread(target=lambda: pipeline.submit([2], request, consume))
            submit_second.start()
            self.assertTrue(second_started.wait(3))
            release_first.set()
            submit_second.join(3)
            self.assertFalse(submit_second.is_alive())
            pipeline.finish(consume)
        finally:
            release_first.set()
            pipeline.close()
        self.assertEqual(consumed, [[1], [2]])

    def test_unresolved_host_has_actionable_error(self):
        args = SimpleNamespace(host="ubuntu-komeiziaya", port=5000)
        error = urllib.error.URLError(socket.gaierror(-3, "Temporary failure in name resolution"))
        with patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(ClientCompatibilityError, "-H の綴り"):
                align_client_model(args)

    def test_batch_http_roundtrip(self):
        class Runtime:
            batch_limit = 4
            metadata = SimpleNamespace(summary=lambda: {
                "protocol": 1, "model_id": "test/model", "metadata_version": "v1", "output_size": 2,
            })

            def predict_images(self, images):
                return [np.asarray([image.getpixel((0, 0))[0] / 255, 0], dtype=np.float32) for image in images]

        with tempfile.TemporaryDirectory() as directory, patch.object(TagServerHandler, "runtime", Runtime()):
            files = [Path(directory) / f"{index}.png" for index in range(2)]
            for path, color in zip(files, ((255, 0, 0), (0, 0, 255))):
                Image.new("RGB", (2, 2), color).save(path)
            server = ParallelTagServer(("127.0.0.1", 0), TagServerHandler, 2)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                metadata = SimpleNamespace(repo_id="test/model", metadata_version="v1", label_count=2)
                result = client_predict_batch(
                    f"http://127.0.0.1:{server.server_port}", [str(path) for path in files], metadata, 5
                )
                np.testing.assert_array_equal(result, [[1, 0], [0, 0]])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_corrupt_image_does_not_fail_batch(self):
        class Runtime:
            batch_limit = 4
            metadata = SimpleNamespace(summary=lambda: {
                "protocol": 1, "model_id": "test/model", "metadata_version": "v1", "output_size": 2,
            })

            def __init__(self):
                self.calls = []

            def predict_images(self, images):
                self.calls.append(len(images))
                return [np.asarray([0.25, 0.75], dtype=np.float32) for _ in images]

        runtime = Runtime()
        with tempfile.TemporaryDirectory() as directory, patch.object(TagServerHandler, "runtime", runtime):
            good = Path(directory) / "good.png"
            bad = Path(directory) / "bad.png"
            Image.new("RGB", (2, 2)).save(good)
            bad.write_bytes(b"\0" * 128)
            server = ParallelTagServer(("127.0.0.1", 0), TagServerHandler, 2)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                metadata = SimpleNamespace(repo_id="test/model", metadata_version="v1", label_count=2)
                url = f"http://127.0.0.1:{server.server_port}"
                result = client_predict_batch(url, [str(good), str(bad)], metadata, 5)
                self.assertIsInstance(result, PartialBatchPredictions)
                np.testing.assert_array_equal(result.predictions[0], [0.25, 0.75])
                self.assertIsNone(result.predictions[1])
                self.assertTrue(result.errors[1])
                all_bad = client_predict_batch(url, [str(bad)], metadata, 5)
                self.assertIsNone(all_bad.predictions[0])
            finally:
                server.shutdown()
                server.server_close()
                thread.join()
        self.assertEqual(runtime.calls, [1])

    def test_tensor_batch_skips_corrupt_image_before_upload(self):
        metadata = SimpleNamespace(repo_id="test/model", metadata_version="v1", label_count=2)
        preprocessor = DBV4Preprocessor({"test": [{"type": "Resize", "size": [8, 8]}]})
        payload = {"protocol": 1, "model_id": "test/model", "metadata_version": "v1",
                   "output_size": 2, "probabilities": [[0.25, 0.75]]}

        class Response:
            headers = {}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

        with tempfile.TemporaryDirectory() as directory:
            good = Path(directory) / "good.png"
            bad = Path(directory) / "bad.png"
            Image.new("RGB", (2, 2)).save(good)
            bad.write_bytes(b"\0" * 128)
            with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
                result = client_predict_tensor_batch(
                    "http://server:5000", [str(bad), str(good)], metadata, preprocessor, 5
                )
                self.assertIsInstance(result, PartialBatchPredictions)
                self.assertIsNone(result.predictions[0])
                np.testing.assert_array_equal(result.predictions[1], [0.25, 0.75])
                self.assertEqual(json.loads(urlopen.call_args.args[0].headers["X-tensor-shape"])[0], 1)
                urlopen.reset_mock()
                all_bad = client_predict_tensor_batch(
                    "http://server:5000", [str(bad)], metadata, preprocessor, 5
                )
                self.assertIsNone(all_bad.predictions[0])
                urlopen.assert_not_called()

    def test_server_batch_preserves_image_order(self):
        image_bytes = []
        for color in ((255, 0, 0), (0, 0, 255)):
            buffer = io.BytesIO()
            Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
            image_bytes.append(base64.b64encode(buffer.getvalue()).decode("ascii"))

        class Runtime:
            batch_limit = None
            metadata = SimpleNamespace(summary=lambda: {
                "protocol": 1, "model_id": "test/model", "metadata_version": "v1", "output_size": 2,
            })

            def predict_images(self, images):
                return [np.asarray([image.getpixel((0, 0))[0] / 255, 0], dtype=np.float32) for image in images]

        handler = TagServerHandler.__new__(TagServerHandler)
        handler.server = SimpleNamespace(_worker_slots=threading.BoundedSemaphore(2))
        handler.runtime = Runtime()
        handler.path = "/batch"
        handler.client_address = ("127.0.0.1", 1000)
        handler.headers = {"Content-Length": "0"}
        body = json.dumps({"images": image_bytes}).encode("utf-8")
        handler.headers["Content-Length"] = str(len(body))
        handler.rfile = io.BytesIO(body)
        responses = []
        handler._send_json_response = lambda status, data: responses.append((status, json.loads(data))) or True
        handler.do_POST()
        self.assertEqual(responses[0][0], 200)
        self.assertEqual(responses[0][1]["probabilities"], [[1.0, 0.0], [0.0, 0.0]])

    def test_client_batch_validates_response_shape(self):
        metadata = SimpleNamespace(repo_id="test/model", metadata_version="v1", label_count=2)
        payload = {
            "protocol": 1, "model_id": "test/model", "metadata_version": "v1",
            "output_size": 2, "probabilities": [[0.1, 0.9], [0.7, 0.3]],
        }

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return json.dumps(payload).encode("utf-8")

            headers = {}

        with tempfile.TemporaryDirectory() as directory:
            files = [Path(directory) / f"{index}.png" for index in range(2)]
            for path in files:
                path.write_bytes(b"image data")
            with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
                result = client_predict_batch("http://server:5000", [str(path) for path in files], metadata, 15)
            self.assertEqual(result.shape, (2, 2))
            request = urlopen.call_args.args[0]
            self.assertEqual(request.full_url, "http://server:5000/batch")
            self.assertEqual(len(json.loads(request.data)["images"]), 2)

    def test_client_batch_accepts_compressed_response(self):
        metadata = SimpleNamespace(repo_id="test/model", metadata_version="v1", label_count=2)
        payload = {"protocol": 1, "model_id": "test/model", "metadata_version": "v1",
                   "output_size": 2, "probabilities": [[0.25, 0.75]]}

        class Response:
            headers = {"Content-Encoding": "gzip"}

            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def read(self):
                return gzip.compress(json.dumps(payload).encode("utf-8"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "image.png"
            path.write_bytes(b"image data")
            with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
                result = client_predict_batch("http://server:5000", [str(path)], metadata, 15)
        np.testing.assert_array_equal(result, [[0.25, 0.75]])
        self.assertEqual(urlopen.call_args.args[0].get_header("Accept-encoding"), "gzip")

    def test_server_compresses_large_opted_in_response(self):
        handler = TagServerHandler.__new__(TagServerHandler)
        handler.headers = {"Accept-Encoding": "gzip"}
        handler.wfile = io.BytesIO()
        headers = {}
        handler.send_response = lambda status: None
        handler.send_header = lambda key, value: headers.__setitem__(key, value)
        handler.end_headers = lambda: None
        body = json.dumps({"probabilities": [0.123456789] * 1000}).encode("utf-8")
        self.assertTrue(handler._send_json_response(200, body))
        self.assertEqual(headers["Content-Encoding"], "gzip")
        self.assertEqual(int(headers["Content-Length"]), len(handler.wfile.getvalue()))
        self.assertEqual(gzip.decompress(handler.wfile.getvalue()), body)


if __name__ == "__main__":
    unittest.main()
