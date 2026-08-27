import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from collect_reviews import SCHEMA, parse_window, stable_review_id, write_csv
from prepare_dataset import clean_and_filter, stratified_sample


def row(review_id, text, platform, app, reviewed_at, rating="5"):
    values = {field: "" for field in SCHEMA}
    values.update(
        {
            "review_id": review_id,
            "review_text": text,
            "review_rating": rating,
            "review_datetime_utc": reviewed_at,
            "application_name": app,
            "application_id": app.lower().replace(" ", "."),
            "platform": platform,
            "store_country": "gb",
            "store_language": "en",
            "developer_response_present": "False",
            "collection_timestamp_utc": "2026-07-28T00:00:00Z",
            "source_url": "https://example.invalid/app",
        }
    )
    return values


class PipelineTests(unittest.TestCase):
    def test_window_validation(self):
        window = parse_window("2025-07-29", "2026-07-28")
        self.assertLess(window.start, window.end)

    def test_stable_identifier(self):
        reviewed_at = datetime(2026, 1, 1, tzinfo=timezone.utc)
        first = stable_review_id("ios", "123", "abc", "Good app", reviewed_at, "salt")
        second = stable_review_id("ios", "123", "abc", "Good app", reviewed_at, "salt")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 32)

    def test_cleaning_removes_duplicate_and_invalid_rating(self):
        rows = [
            row("a", "The app works very well for payments.", "android", "Monzo", "2026-01-01T00:00:00Z"),
            row("b", "The app works very well for payments.", "android", "Monzo", "2026-01-01T01:00:00Z"),
            row("c", "The login process is unreliable.", "ios", "Monzo", "2026-01-02T00:00:00Z", "8"),
        ]
        cleaned, reasons = clean_and_filter(
            rows,
            datetime(2025, 7, 29, tzinfo=timezone.utc),
            datetime(2026, 7, 28, 23, 59, tzinfo=timezone.utc),
            False,
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(reasons["duplicate_same_app_platform_date_text"], 1)
        self.assertEqual(reasons["invalid_rating"], 1)

    def test_stratified_sample_uses_both_platforms(self):
        rows = []
        for index in range(10):
            rows.append(row(f"a{index}", f"Android review {index}", "android", "Monzo", "2026-01-01T00:00:00Z"))
            rows.append(row(f"i{index}", f"iOS review {index}", "ios", "Monzo", "2026-01-01T00:00:00Z"))
        selected = stratified_sample(rows, 6, 17686)
        self.assertEqual(len(selected), 6)
        self.assertEqual({item["platform"] for item in selected}, {"android", "ios"})

    def test_template_write_has_all_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "template.csv"
            write_csv(path, [])
            header = path.read_text(encoding="utf-8-sig").splitlines()[0]
            self.assertEqual(header.split(","), SCHEMA)


if __name__ == "__main__":
    unittest.main()
