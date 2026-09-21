import unittest

from make_report import get_badge_info


class ReportBadgeTests(unittest.TestCase):
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
