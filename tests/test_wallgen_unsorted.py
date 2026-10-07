import os
from pathlib import Path
import tempfile
import unittest

from embed_tags_universal import (
    APP_CONFIG,
    collect_wallgen_unsorted_image_groups,
    folder_name_for_rating,
    get_wallgen_unsorted_move_rating,
    inference_timing_summary,
    organize_wallgen_unsorted_folder,
)


class WallgenUnsortedOrganizationTests(unittest.TestCase):
    def test_folder_name_falls_back_to_base_rating(self):
        original = APP_CONFIG["folder_names"]
        APP_CONFIG["folder_names"] = {"questionable": "R-17"}
        try:
            self.assertEqual(folder_name_for_rating("questionable_3"), "R-17")
        finally:
            APP_CONFIG["folder_names"] = original

    def test_inference_timing_excludes_slow_first_batch(self):
        summary = inference_timing_summary(
            8,
            4.4,
            [
                {"count": 4, "time": 4.0},
                {"count": 4, "time": 0.4},
            ],
        )
        self.assertTrue(summary["outlier_detected"])
        self.assertEqual(summary["main_count"], 4)
        self.assertAlmostEqual(summary["fps"], 10.0)

    def test_collects_images_from_leaf_directories_only(self):
        with tempfile.TemporaryDirectory() as directory:
            child = os.path.join(directory, "artist")
            excluded = os.path.join(
                directory,
                APP_CONFIG["folder_names"]["questionable_1"],
            )
            os.makedirs(child)
            os.makedirs(excluded)
            root_image = os.path.join(directory, "root.jpg")
            child_image = os.path.join(child, "child.png")
            excluded_image = os.path.join(excluded, "already_moved.webp")
            for path in (root_image, child_image, excluded_image):
                with open(path, "wb") as file:
                    file.write(b"test")

            groups = collect_wallgen_unsorted_image_groups([directory])

            self.assertNotIn(os.path.abspath(directory), groups)
            self.assertEqual(groups[os.path.abspath(child)], [child_image])
            self.assertNotIn(os.path.abspath(excluded), groups)

    def test_default_and_custom_move_minimum(self):
        self.assertEqual(get_wallgen_unsorted_move_rating(["general", "sensitive_0"]), "sensitive_0")
        self.assertIsNone(get_wallgen_unsorted_move_rating(["general"]))
        self.assertIsNone(get_wallgen_unsorted_move_rating(["sensitive_4"], "R-17_0"))
        self.assertEqual(get_wallgen_unsorted_move_rating(["sensitive_4", "questionable_0"], "R-17_0"), "questionable_0")
        self.assertIsNone(get_wallgen_unsorted_move_rating(["questionable_4"], "R-18"))
        self.assertEqual(get_wallgen_unsorted_move_rating(["explicit"], "R-18"), "explicit")
        with self.assertRaises(ValueError):
            get_wallgen_unsorted_move_rating(["explicit"], "invalid")

    def test_moves_sensitive_group_and_excludes_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            unsorted = os.path.join(directory, "未整理")
            source = os.path.join(unsorted, "作者")
            os.makedirs(source)
            paths = [os.path.join(source, name) for name in ("one.jpg", "two.png")]
            for path in paths:
                Path(path).write_bytes(b"test")
            moved, count = organize_wallgen_unsorted_folder(paths, "sensitive_0", [source])
            self.assertEqual(count, 2)
            self.assertTrue(all(os.path.dirname(path) == source + "_R-15_0" for path in moved.values()))
            self.assertEqual(collect_wallgen_unsorted_image_groups([unsorted]), {})
            self.assertTrue(os.path.isdir(unsorted))

    def test_moves_whole_group_for_questionable_rating(self):
        with tempfile.TemporaryDirectory() as directory:
            source_root = os.path.join(directory, "source")
            artist = os.path.join(source_root, "artist")
            os.makedirs(artist)
            files = [
                os.path.join(artist, "one.jpg"),
                os.path.join(artist, "two.png"),
            ]
            for path in files:
                with open(path, "wb") as file:
                    file.write(b"test")

            rating = get_wallgen_unsorted_move_rating(["general", "questionable_2"])
            moved_paths, moved_count = organize_wallgen_unsorted_folder(
                files,
                rating,
                [source_root],
            )

            target_root = os.path.join(
                directory,
                "source_" + APP_CONFIG["folder_names"]["questionable_2"],
                "artist",
            )
            self.assertEqual(rating, "questionable_2")
            self.assertEqual(moved_count, 2)
            self.assertEqual(len(moved_paths), 2)
            self.assertTrue(os.path.isfile(os.path.join(target_root, "one.jpg")))
            self.assertTrue(os.path.isfile(os.path.join(target_root, "two.png")))
            self.assertFalse(os.path.exists(artist))


if __name__ == "__main__":
    unittest.main()
