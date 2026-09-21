import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from PIL import Image

import embed_tags_universal as app
from dbv4 import DBV4Metadata


class RuntimeCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _metadata():
        labels = ("general", "sensitive", "questionable", "explicit", "tag_a")
        return DBV4Metadata(
            repo_id="test/model",
            profile_name="test",
            model_path="",
            tags_path="",
            preprocess_path="",
            categories_path=None,
            thresholds_path=None,
            labels=labels,
            categories=("rating", "rating", "rating", "rating", "general"),
            tag_thresholds={name: 0.5 for name in labels},
            rating_indices={
                "general": 0,
                "sensitive": 1,
                "questionable": 2,
                "explicit": 3,
            },
            metadata_version="test-version",
        )

    def test_migrates_known_legacy_profiles_without_removing_custom_profiles(self):
        config = {
            "model_profile": "large",
            "model_profiles": {
                "lightweight": {"repo_id": "animetimm/caformer_m36.dbv4-full"},
                "balanced": {"repo_id": "animetimm/convformer_s36.dbv4-full"},
                "high": {"repo_id": "animetimm/swinv2_base_window8_256.dbv4-full"},
                "large": {"repo_id": "animetimm/eva02_large_patch14_448.dbv4-full"},
                "ultra": {"repo_id": "animetimm/convnextv2_huge.dbv4-full"},
                "custom": {"repo_id": "example/custom.dbv4-full"},
            },
        }

        changed = app.migrate_legacy_config(config)

        self.assertTrue(changed)
        self.assertEqual(config["model_profile"], "high")
        self.assertEqual(
            config["model_profiles"]["balanced"]["repo_id"],
            "animetimm/caformer_b36.dbv4-full",
        )
        self.assertEqual(
            config["model_profiles"]["custom"]["repo_id"],
            "example/custom.dbv4-full",
        )
        self.assertEqual(
            config["model_profiles"]["ultra"]["repo_id"],
            "itterative/convnextv2_huge.dbv4-full-onnx",
        )
        self.assertEqual(
            config["model_profiles"]["ultra"]["metadata_repo_id"],
            "animetimm/convnextv2_huge.dbv4-full",
        )
        self.assertEqual(
            config["model_profiles"]["ultra"]["model_external_files"],
            ["model.onnx_data"],
        )
        self.assertIn(
            "5,743MiB",
            config["model_profiles"]["ultra"]["vram_warning"],
        )

    def test_parser_accepts_custom_model_profile(self):
        args = app.create_parser().parse_args(["--model-profile", "custom", "image.jpg"])
        self.assertEqual(args.model_profile, "custom")

    def test_build_providers_skips_tensorrt_when_runtime_is_missing(self):
        available = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        with (
            patch.object(app, "IS_WINDOWS", False),
            patch.object(app, "IS_LINUX", True),
            patch.object(app.ort, "get_available_providers", return_value=available),
            patch.object(app.ctypes, "CDLL", side_effect=OSError("missing")),
        ):
            providers = app.build_providers(True)
        self.assertEqual(providers, ["CUDAExecutionProvider", "CPUExecutionProvider"])

    def test_build_providers_uses_tensorrt_when_runtime_is_loadable(self):
        available = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        with (
            patch.object(app, "IS_WINDOWS", False),
            patch.object(app, "IS_LINUX", True),
            patch.object(app.ort, "get_available_providers", return_value=available),
            patch.object(app.ctypes, "CDLL"),
        ):
            providers = app.build_providers(True)
        self.assertEqual(
            providers,
            ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def test_manual_approval_profiles_are_available(self):
        compact = app.MODEL_PROFILES["compact_manual"]
        medium = app.MODEL_PROFILES["medium_manual"]
        self.assertEqual(compact["repo_id"], "animetimm/repvit_m2_3.dbv4-full")
        self.assertEqual(medium["repo_id"], "animetimm/convformer_s36.dbv4-full")
        self.assertIn("管理者", compact["access_notice"])
        self.assertIn("管理者", medium["access_notice"])

    def test_future_1b_profile_is_reserved_until_onnx_is_available(self):
        profile = app.MODEL_PROFILES["future_1b"]
        self.assertFalse(profile["available"])
        self.assertEqual(
            profile["repo_id"],
            "animetimm/vit_giantopt_patch16_siglip_384.dbv4-full",
        )
        self.assertIn("ONNX", profile["unavailable_reason"])

    def test_manual_profile_starts_browser_login_when_logged_out(self):
        profile = app.MODEL_PROFILES["compact_manual"]
        with (
            patch.object(app, "get_token", return_value=None),
            patch.object(app, "login") as login,
            patch.object(app, "get_hf_file_metadata") as metadata,
        ):
            app.ensure_profile_access("compact_manual", profile)

        login.assert_called_once_with(skip_if_logged_in=True)
        metadata.assert_called_once()

    def test_manual_profile_opens_approval_page_when_access_is_denied(self):
        profile = app.MODEL_PROFILES["medium_manual"]
        request = httpx.Request("HEAD", "https://huggingface.co/test/model.onnx")
        response = httpx.Response(403, request=request)
        denied = app.GatedRepoError("approval required", response=response)
        with (
            patch.object(app, "get_token", return_value="saved-token"),
            patch.object(app, "get_hf_file_metadata", side_effect=denied),
            patch.object(app.webbrowser, "open", return_value=True) as browser,
            self.assertRaises(SystemExit),
        ):
            app.ensure_profile_access("medium_manual", profile)

        browser.assert_called_once_with(
            "https://huggingface.co/animetimm/convformer_s36.dbv4-full",
            new=2,
        )

    def test_client_http_error_skips_only_failed_image(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [f"{directory}/one.jpg", f"{directory}/two.jpg"]
            args = SimpleNamespace(
                mode="client",
                gpu=False,
                model_profile="test",
                model_repo=None,
                model_file=None,
                tags_file=None,
                no_tag=True,
                organize=False,
                record_ratio=None,
                pixiv=False,
                recursive=False,
                images=paths,
                batch_size=4,
                io_workers=-1,
                force=True,
                rating_thresh=None,
                ignore_sensitive=False,
                thresh=None,
                host="localhost",
                port=5000,
                no_report=True,
            )
            prediction = np.array([0.9, 0.1, 0.05, 0.01, 0.8], dtype=np.float32)
            http_error = urllib.error.HTTPError(
                "http://localhost:5000",
                500,
                "server error",
                None,
                None,
            )

            with (
                patch.object(app, "load_client_metadata", return_value=self._metadata()),
                patch.object(app, "collect_images", return_value=paths),
                patch.object(app, "client_predict", side_effect=[http_error, prediction]) as predict,
            ):
                app.process_images(args)

            self.assertEqual(predict.call_count, 2)

    def test_client_compatibility_error_stops_remaining_images(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [f"{directory}/one.jpg", f"{directory}/two.jpg"]
            args = SimpleNamespace(
                mode="client",
                gpu=False,
                model_profile="test",
                model_repo=None,
                model_file=None,
                tags_file=None,
                no_tag=True,
                organize=False,
                record_ratio=None,
                pixiv=False,
                recursive=False,
                images=paths,
                batch_size=4,
                io_workers=-1,
                force=True,
                rating_thresh=None,
                ignore_sensitive=False,
                thresh=None,
                host="localhost",
                port=5000,
                no_report=True,
            )

            with self.assertRaises(SystemExit) as raised:
                with (
                    patch.object(app, "load_client_metadata", return_value=self._metadata()),
                    patch.object(app, "collect_images", return_value=paths),
                    patch.object(
                        app,
                        "client_predict",
                        side_effect=app.ClientCompatibilityError("model_id mismatch"),
                    ) as predict,
                ):
                    app.process_images(args)

            self.assertEqual(raised.exception.code, 1)
            self.assertEqual(predict.call_count, 1)

    def test_server_disconnected_client_does_not_raise_broken_pipe(self):
        handler = object.__new__(app.TagServerHandler)
        handler.send_response = lambda status: None
        handler.send_header = lambda name, value: None
        handler.end_headers = lambda: None
        handler.wfile = SimpleNamespace(write=lambda body: (_ for _ in ()).throw(BrokenPipeError()))
        self.assertFalse(handler._send_json_response(200, b"{}"))

    def test_parallel_server_limits_active_requests(self):
        active = 0
        maximum = 0
        lock = threading.Lock()

        class SlowHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                nonlocal active, maximum
                with lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.1)
                with lock:
                    active -= 1
                self.send_response(200)
                self.end_headers()

            def log_message(self, format, *args):
                return

        server = app.ParallelTagServer(("127.0.0.1", 0), SlowHandler, 2)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}"
            with ThreadPoolExecutor(max_workers=4) as executor:
                results = list(executor.map(lambda _: urllib.request.urlopen(url).status, range(4)))
            self.assertEqual(results, [200] * 4)
            self.assertEqual(maximum, 2)
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_postprocess_error_does_not_retry_batch_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [f"{directory}/one.jpg", f"{directory}/two.jpg"]
            for path in paths:
                Image.new("RGB", (8, 8), (128, 128, 128)).save(path)
            prediction = np.array([0.9, 0.1, 0.05, 0.01, 0.8], dtype=np.float32)
            predict_calls = []

            def predict(images):
                predict_calls.append(len(images))
                return [prediction.copy() for _ in images]

            runtime = SimpleNamespace(
                metadata=self._metadata(),
                batch_limit=None,
                predict_images=predict,
            )
            args = SimpleNamespace(
                mode="standalone",
                gpu=False,
                model_profile="test",
                model_repo=None,
                model_file=None,
                tags_file=None,
                no_tag=True,
                organize=False,
                record_ratio=None,
                pixiv=False,
                recursive=False,
                images=paths,
                batch_size=2,
                io_workers=0,
                force=True,
                rating_thresh=None,
                ignore_sensitive=False,
                thresh=None,
                host="localhost",
                port=5000,
                no_report=True,
            )

            with (
                patch.object(app, "load_runtime_model", return_value=runtime),
                patch.object(app, "warmup_runtime", return_value=0.0),
                patch.object(app, "collect_images", return_value=paths),
                patch.object(app, "calculate_rating", side_effect=RuntimeError("decode failed")),
            ):
                app.process_images(args)

            self.assertEqual(predict_calls, [2])


if __name__ == "__main__":
    unittest.main()
