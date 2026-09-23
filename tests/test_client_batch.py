import base64
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

import numpy as np
from PIL import Image

from embed_tags_universal import (
    ClientCompatibilityError,
    ParallelTagServer,
    TagServerHandler,
    align_client_model,
    client_predict_batch,
)


class BatchProtocolTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
