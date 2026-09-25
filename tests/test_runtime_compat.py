import tempfile
import socket
import threading
import time
import unittest
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
from PIL import Image

import embed_tags_universal as app
from dbv4 import DBV4Metadata


class RuntimeCompatibilityTests(unittest.TestCase):
    def test_exiftool_diagnostics_are_not_merged_into_tag_output(self):
        process = MagicMock()
        wrapper = app.ExifToolWrapper("exiftool")

        with patch.object(app.subprocess, "Popen", return_value=process) as popen:
            wrapper.start()

        self.assertIsNone(popen.call_args.kwargs["stderr"])
        self.assertIsNot(popen.call_args.kwargs["stderr"], app.subprocess.STDOUT)
        wrapper.stop()

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
            patch.object(app.gpu_runtime, "prepare_tensorrt", side_effect=RuntimeError("missing")),
            patch.object(app.gpu_runtime, "prepare_cuda"),
        ):
            providers = app.build_providers(True)
        self.assertEqual(providers, [("CUDAExecutionProvider", {"device_id": "0"}), "CPUExecutionProvider"])

    def test_build_providers_uses_tensorrt_when_runtime_is_loadable(self):
        available = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        with (
            patch.object(app, "IS_WINDOWS", False),
            patch.object(app, "IS_LINUX", True),
            patch.object(app.ort, "get_available_providers", return_value=available),
            patch.object(app.gpu_runtime, "prepare_tensorrt"),
            patch.object(app.gpu_runtime, "prepare_cuda"),
        ):
            providers = app.build_providers(True)
        self.assertEqual(
            providers,
            [
                ("TensorrtExecutionProvider", app.gpu_runtime.tensorrt_provider_options(0)),
                ("CUDAExecutionProvider", {"device_id": "0"}),
                "CPUExecutionProvider",
            ],
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
            patch.object(app, "login") as login,
            patch.object(app, "get_hf_file_metadata", side_effect=denied),
            patch.object(app.webbrowser, "open", return_value=True) as browser,
            self.assertRaises(SystemExit),
        ):
            app.ensure_profile_access("medium_manual", profile)

        login.assert_called_once_with(skip_if_logged_in=False)
        browser.assert_called_once_with(
            "https://huggingface.co/animetimm/convformer_s36.dbv4-full",
            new=2,
        )

    def test_gated_profile_retries_after_login(self):
        profile = app.MODEL_PROFILES["ultra"]
        request = httpx.Request("HEAD", "https://huggingface.co/test/selected_tags.csv")
        denied = app.GatedRepoError("login required", response=httpx.Response(401, request=request))
        with (
            patch.object(app, "get_token", return_value="expired-token"),
            patch.object(app, "login") as login,
            patch.object(app, "get_hf_file_metadata", side_effect=[denied, None]) as metadata,
            patch.object(app.webbrowser, "open", return_value=True),
        ):
            app.ensure_profile_access("ultra", profile)
        login.assert_called_once_with(skip_if_logged_in=False)
        self.assertEqual(metadata.call_count, 2)

    def test_ultra_checks_gated_metadata_before_model_download(self):
        profile = app.MODEL_PROFILES["ultra"]
        with (
            patch.object(app, "get_token", return_value="saved-token"),
            patch.object(app, "hf_hub_url", return_value="metadata-url") as hub_url,
            patch.object(app, "get_hf_file_metadata") as metadata,
        ):
            app.ensure_profile_access("ultra", profile)
        hub_url.assert_called_once_with(
            repo_id="animetimm/convnextv2_huge.dbv4-full",
            filename="selected_tags.csv",
        )
        metadata.assert_called_once_with("metadata-url", token=True)

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
                client_upload_mode="original",
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
                patch.object(app, "align_client_model", return_value=self._metadata()),
                patch.object(app, "collect_images", return_value=paths),
                patch.object(app, "client_predict", side_effect=[http_error, prediction]) as predict,
            ):
                app.process_images(args)

            self.assertEqual(predict.call_count, 2)

    def test_client_auto_selects_server_profile_without_model_download(self):
        metadata = replace(
            self._metadata(), repo_id="animetimm/mobilenetv4_conv_aa_large.dbv4-full"
        )
        payload = {
            "protocol": 1,
            "model_id": metadata.repo_id,
            "profile": "lightweight",
            "metadata_version": metadata.metadata_version,
            "output_size": metadata.label_count,
        }
        response = MagicMock()
        response.__enter__.return_value.read.return_value = app.json.dumps(payload).encode("utf-8")
        args = SimpleNamespace(
            host="127.0.0.1", port=5000, model_profile="balanced",
            model_repo=None, auto_model_profile=True,
        )
        with (
            patch.object(app.urllib.request, "urlopen", return_value=response),
            patch.object(app, "load_client_metadata", return_value=metadata) as load,
        ):
            result = app.align_client_model(args)
        self.assertIs(result, metadata)
        self.assertEqual(args.model_profile, "lightweight")
        load.assert_called_once_with(args)

    def test_client_waits_for_server_model_loading(self):
        metadata = self._metadata()
        loading = urllib.error.HTTPError(
            "http://server/metadata", 503, "loading", {},
            app.io.BytesIO(b'{"status":"loading","stage":"TensorRT engine build"}'),
        )
        response = MagicMock()
        response.__enter__.return_value.read.return_value = app.json.dumps(metadata.summary()).encode()
        args = SimpleNamespace(host="server", port=5000, model_profile="test", model_repo=None,
                               auto_model_profile=False)
        with patch.object(app.urllib.request, "urlopen", side_effect=[loading, response]) as urlopen, \
             patch.object(app.time, "sleep") as sleep, \
             patch.object(app, "load_client_metadata", return_value=metadata):
            self.assertIs(app.align_client_model(args), metadata)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once_with(2)

    def test_client_syncs_preprocessor_packages_in_virtualenv(self):
        server_environment = app.preprocessor_environment() | {"Pillow": "11.3.0", "NumPy": "2.1.3"}
        with patch.object(app.sys, "prefix", "/tmp/client-venv"), \
             patch.object(app.sys, "base_prefix", "/usr"), \
             patch.dict(app.os.environ, {}, clear=True), \
             patch.object(app.subprocess, "run") as install, \
             patch.object(app.os, "execv", side_effect=RuntimeError("restarted")):
            with self.assertRaisesRegex(RuntimeError, "restarted"):
                app.sync_client_preprocessor_packages(server_environment)
        self.assertEqual(install.call_args.args[0][-2:], ["Pillow==11.3.0", "numpy==2.1.3"])

    def test_client_reports_server_startup_failure(self):
        failed = urllib.error.HTTPError(
            "http://server/metadata", 503, "loading", {},
            app.io.BytesIO(b'{"status":"error","error":"TensorRT initialization failed"}'),
        )
        args = SimpleNamespace(host="server", port=5000)
        with patch.object(app.urllib.request, "urlopen", side_effect=failed):
            with self.assertRaisesRegex(app.ClientCompatibilityError, "TensorRT initialization failed"):
                app.align_client_model(args)

    def test_incompatible_preprocessing_falls_back_to_original_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/one.jpg"
            args = SimpleNamespace(
                mode="client", gpu=False, model_profile="test", model_repo=None,
                model_file=None, tags_file=None, no_tag=True, organize=False,
                record_ratio=None, pixiv=False, recursive=False, images=[path],
                batch_size=1, io_workers=0, force=True, rating_thresh=None,
                ignore_sensitive=False, thresh=None, host="localhost", port=5000,
                no_report=True, client_upload_mode="preprocessed",
                client_tensor_supported=True, client_tensor_hash="server-hash",
                client_tensor_environment=app.preprocessor_environment(),
            )
            prediction = np.array([0.9, 0.1, 0.05, 0.01, 0.8], dtype=np.float32)
            with patch.object(app, "align_client_model", return_value=self._metadata()), \
                 patch.object(app, "collect_images", return_value=[path]), \
                 patch.object(app.DBV4Preprocessor, "from_metadata", return_value=MagicMock()), \
                 patch.object(app, "preprocessor_probe_hash", return_value="client-hash"), \
                 patch.object(app, "sync_client_preprocessor_packages",
                              side_effect=app.ClientCompatibilityError("sync failed")), \
                 patch.object(app, "client_predict", return_value=prediction) as original, \
                 patch.object(app, "client_predict_tensor_batch") as tensor:
                app.process_images(args)
            original.assert_called_once()
            tensor.assert_not_called()

    def test_tensor_request_failure_switches_to_original_upload(self):
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/one.jpg"
            args = SimpleNamespace(
                mode="client", gpu=False, model_profile="test", model_repo=None,
                model_file=None, tags_file=None, no_tag=True, organize=False,
                record_ratio=None, pixiv=False, recursive=False, images=[path],
                batch_size=1, io_workers=0, force=True, rating_thresh=None,
                ignore_sensitive=False, thresh=None, host="localhost", port=5000,
                no_report=True, client_upload_mode="preprocessed",
                client_tensor_supported=True, client_tensor_hash="same",
            )
            prediction = np.array([0.9, 0.1, 0.05, 0.01, 0.8], dtype=np.float32)
            with patch.object(app, "align_client_model", return_value=self._metadata()), \
                 patch.object(app, "collect_images", return_value=[path]), \
                 patch.object(app.DBV4Preprocessor, "from_metadata", return_value=MagicMock()), \
                 patch.object(app, "preprocessor_probe_hash", return_value="same"), \
                 patch.object(app, "client_predict_tensor_batch", side_effect=ValueError("tensor rejected")), \
                 patch.object(app, "client_predict", return_value=prediction) as original:
                app.process_images(args)
            original.assert_called_once()

    def test_client_batch_timeout_retries_singles_and_keeps_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [f"{directory}/{index}.jpg" for index in range(3)]
            args = SimpleNamespace(
                mode="client", gpu=False, model_profile="test", model_repo=None,
                model_file=None, tags_file=None, no_tag=True, organize=False,
                record_ratio=None, pixiv=False, recursive=False, images=paths,
                batch_size=2, io_workers=0, force=True, rating_thresh=None,
                ignore_sensitive=False, thresh=None, host="localhost", port=5000,
                no_report=True, client_batch_supported=True, client_batch_limit=None,
                client_upload_mode="original",
            )
            prediction = np.array([0.9, 0.1, 0.05, 0.01, 0.8], dtype=np.float32)
            with (
                patch.object(app, "align_client_model", return_value=self._metadata()),
                patch.object(app, "collect_images", return_value=paths),
                patch.object(app, "client_predict_batch", side_effect=socket.timeout) as batch,
                patch.object(app, "client_predict", return_value=prediction) as single,
            ):
                app.process_images(args)
            self.assertEqual(batch.call_count, 2)
            self.assertTrue(all(call.args[-1] == 120 for call in batch.call_args_list))
            self.assertEqual(single.call_count, 3)

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
                client_upload_mode="original",
            )

            with self.assertRaises(SystemExit) as raised:
                with (
                    patch.object(app, "align_client_model", return_value=self._metadata()),
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

    def test_server_metadata_endpoint_returns_model_without_inference(self):
        metadata = self._metadata()
        runtime = SimpleNamespace(metadata=metadata)
        with patch.object(app.TagServerHandler, "runtime", runtime):
            server = app.ParallelTagServer(("127.0.0.1", 0), app.TagServerHandler, 2)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                url = f"http://127.0.0.1:{server.server_port}/metadata"
                with urllib.request.urlopen(url) as response:
                    payload = app.json.loads(response.read().decode("utf-8"))
                self.assertEqual({key: payload[key] for key in metadata.summary()}, metadata.summary())
                self.assertTrue(payload["batch_supported"])
                self.assertEqual(payload["batch_limit"], 8)
                self.assertNotIn("probabilities", payload)
            finally:
                server.shutdown()
                server.server_close()
                thread.join()

    def test_server_binds_before_loading_model(self):
        loading = threading.Event()
        release = threading.Event()
        servers = []
        real_server = app.ParallelTagServer
        metadata = self._metadata()
        runtime = SimpleNamespace(metadata=metadata, batch_limit=None)

        def create_server(*args):
            server = real_server(*args)
            servers.append(server)
            return server

        def load_model(*args, **kwargs):
            loading.set()
            self.assertTrue(release.wait(3))
            return runtime

        args = SimpleNamespace(gpu=True, model_profile="test", model_repo=None,
                               model_file=None, tags_file=None, webgpu=False, port=0)
        with patch.object(app, "ParallelTagServer", side_effect=create_server), \
             patch.object(app, "load_runtime_model", side_effect=load_model):
            thread = threading.Thread(target=app.run_server, args=(args,), daemon=True)
            thread.start()
            try:
                self.assertTrue(loading.wait(3))
                url = f"http://127.0.0.1:{servers[0].server_port}/metadata"
                with self.assertRaises(urllib.error.HTTPError) as raised:
                    urllib.request.urlopen(url, timeout=3)
                self.assertEqual(raised.exception.code, 503)
                self.assertEqual(app.json.load(raised.exception)["status"], "loading")
                release.set()
                for _ in range(30):
                    try:
                        with urllib.request.urlopen(url, timeout=3) as response:
                            self.assertEqual(app.json.load(response)["model_id"], metadata.repo_id)
                        break
                    except urllib.error.HTTPError as exc:
                        self.assertEqual(exc.code, 503)
                        time.sleep(0.05)
                else:
                    self.fail("Server did not become ready")
            finally:
                release.set()
                if servers:
                    servers[0].shutdown()
                thread.join(3)
            self.assertFalse(thread.is_alive())

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
