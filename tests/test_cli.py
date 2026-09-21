import unittest

from embed_tags_universal import create_parser


class CliShortOptionTests(unittest.TestCase):
    def test_all_python_cli_short_options(self):
        args = create_parser().parse_args([
            "-D", "client",
            "-Z",
            "-o",
            "-x",
            "-z",
            "-q", "0.42",
            "-g",
            "-b", "2",
            "-w", "3",
            "-f",
            "-r",
            "-m", "balanced",
            "-e", "owner/model",
            "-l", "model.onnx",
            "-y", "selected_tags.csv",
            "-j", "127.0.0.1",
            "-u", "5000",
            "-v", "4",
            "-a",
            "-d", "0.5",
            "-i",
            "images",
        ])

        self.assertEqual(args.mode, "client")
        self.assertTrue(args.no_tag)
        self.assertTrue(args.organize)
        self.assertTrue(args.pixiv)
        self.assertTrue(args.no_report)
        self.assertEqual(args.thresh, 0.42)
        self.assertTrue(args.gpu)
        self.assertEqual(args.batch_size, 2)
        self.assertEqual(args.io_workers, 3)
        self.assertTrue(args.force)
        self.assertTrue(args.recursive)
        self.assertEqual(args.model_profile, "balanced")
        self.assertEqual(args.model_repo, "owner/model")
        self.assertEqual(args.model_file, "model.onnx")
        self.assertEqual(args.tags_file, "selected_tags.csv")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 5000)
        self.assertEqual(args.sensitive_split_mode, 4)
        self.assertTrue(args.record_ratio)
        self.assertEqual(args.rating_thresh, 0.5)
        self.assertTrue(args.ignore_sensitive)
        self.assertEqual(args.images, ["images"])

    def test_negative_short_switches(self):
        args = create_parser().parse_args(["-n", "-k", "-G"])
        self.assertFalse(args.recursive)
        self.assertFalse(args.record_ratio)
        self.assertTrue(args.gen_config)


if __name__ == "__main__":
    unittest.main()
