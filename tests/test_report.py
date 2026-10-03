import unittest
import json
import os
import tempfile
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import unquote
from unittest.mock import patch

import make_report

from make_report import get_badge_info


class ReportBadgeTests(unittest.TestCase):
    def test_report_preserves_special_filename_without_html_or_script_injection(self):
        class Parser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.images = []
                self.clicks = []

            def handle_starttag(self, tag, attrs):
                attrs = dict(attrs)
                if tag == 'img':
                    self.images.append(attrs)
                if 'onclick' in attrs:
                    self.clicks.append(attrs['onclick'])

        old_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                filename = 'a\' & # <img src=x onerror=alert(1)>.png'
                log = Path(directory) / 'report_log.json'
                log.write_text(json.dumps([{'path': str(Path(directory) / filename),
                                           'rating': 'general', 'probs': [0.9, 0.1, 0.0, 0.0]}]))
                with patch.object(make_report, 'REPORT_LOG_FILE', str(log)), patch.object(
                        make_report, 'load_config', return_value={'folder_names': {'general': 'R-00'}}):
                    make_report.make_report()
                report = Path('report_R-00.html').read_text()
                parser = Parser()
                parser.feed(report)
                self.assertEqual(len(parser.images), 2)  # card and fixed preview modal
                self.assertEqual(unquote(parser.images[0]['src']), filename)
                self.assertEqual(parser.clicks[0], "show(this.querySelector('img').src)")
                self.assertIn('&lt;img', report)
                self.assertNotIn('onerror', parser.images[0])
            finally:
                os.chdir(old_cwd)

    def test_dbv4_sensitive_badge_uses_sensitive_score(self):
        badge_class, badge_text, score_text = get_badge_info(
            "sensitive_4",
            [0.1, 0.8, 0.2, 0.05],
        )
        self.assertEqual(badge_class, "badge-lvl5")
        self.assertEqual(badge_text, "R-15_4")
        self.assertIn("Sen: 0.8000", score_text)

    def test_dbv4_questionable_badge_uses_questionable_score(self):
        badge_class, badge_text, score_text = get_badge_info(
            "questionable_3",
            [0.1, 0.2, 0.7, 0.05],
        )
        self.assertEqual(badge_class, "badge-lvl5")
        self.assertEqual(badge_text, "R-17_3")
        self.assertIn("Que: 0.7000", score_text)


if __name__ == "__main__":
    unittest.main()
