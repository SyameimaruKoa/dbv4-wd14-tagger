import os
import tempfile
import unittest

from embed_tags_universal import (
    APP_CONFIG,
    collect_pixiv_image_groups,
    folder_name_for_rating,
    get_pixiv_move_rating,
    inference_timing_summary,
    organize_pixiv_folder,
)


class PixivOrganizationTests(unittest.TestCase):
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

    def test_collects_images_from_parent_and_child_directories(self):
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

            groups = collect_pixiv_image_groups([directory])

            self.assertEqual(groups[os.path.abspath(directory)], [root_image])
            self.assertEqual(groups[os.path.abspath(child)], [child_image])
            self.assertNotIn(os.path.abspath(excluded), groups)

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

            rating = get_pixiv_move_rating(["general", "questionable_2"])
            moved_paths, moved_count = organize_pixiv_folder(
                files,
                rating,
                [source_root],
            )

            target_root = os.path.join(
                directory,
                APP_CONFIG["folder_names"]["questionable_2"],
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
