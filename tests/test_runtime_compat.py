import tempfile
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch

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

    def test_parser_accepts_custom_model_profile(self):
        args = app.create_parser().parse_args(["--model-profile", "custom", "image.jpg"])
        self.assertEqual(args.model_profile, "custom")

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
