#!/usr/bin/env python3
"""Collect public Google Play and Apple App Store reviews into one safe schema.

The collector deliberately excludes usernames, author identifiers, profile
images, and developer-response text. It writes only review content and the
minimum metadata required by the dissertation design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import random
import sys
import time
import unicodedata
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, time as datetime_time, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator


SCHEMA = [
    "review_id",
    "review_text",
    "review_title",
    "review_rating",
    "review_datetime_utc",
    "application_name",
    "application_id",
    "platform",
    "store_country",
    "store_language",
    "app_version",
    "review_likes",
    "developer_response_present",
    "collection_timestamp_utc",
    "source_url",
]


@dataclass(frozen=True)
class DateWindow:
    start: datetime
    end: datetime

    def contains(self, value: datetime) -> bool:
        return self.start <= value <= self.end


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect and normalise public mobile-banking app reviews."
    )
    parser.add_argument("--config", default="apps.json", help="Application config JSON.")
    parser.add_argument("--output-dir", default="output", help="Output directory.")
    parser.add_argument(
        "--platform",
        choices=("both", "google", "apple"),
        default="both",
        help="Storefront(s) to collect.",
    )
    parser.add_argument(
        "--date-start",
        required=True,
        help="Inclusive UTC start date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--date-end",
        required=True,
        help="Inclusive UTC end date in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--max-google-per-app",
        type=int,
        default=4000,
        help="Maximum eligible Google reviews retained per application.",
    )
    parser.add_argument(
        "--max-apple-per-app",
        type=int,
        default=500,
        help="Maximum eligible Apple reviews retained per application (public limit is 500).",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=1.0,
        help="Base delay in seconds between storefront requests.",
    )
    parser.add_argument(
        "--review-id-salt",
        default="UC-17686-cross-platform-review",
        help="Non-personal project salt used only to derive stable review IDs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and write empty schema files without network calls.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def parse_window(start_text: str, end_text: str) -> DateWindow:
    start_date = date.fromisoformat(start_text)
    end_date = date.fromisoformat(end_text)
    if start_date > end_date:
        raise ValueError("--date-start must not be later than --date-end")
    return DateWindow(
        start=datetime.combine(start_date, datetime_time.min, tzinfo=timezone.utc),
        end=datetime.combine(end_date, datetime_time.max, tzinfo=timezone.utc),
    )


def load_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    required_top = {"study", "country", "language", "applications"}
    missing_top = required_top.difference(data)
    if missing_top:
        raise ValueError(f"Config missing keys: {sorted(missing_top)}")
    if not isinstance(data["applications"], list) or not data["applications"]:
        raise ValueError("Config must contain at least one application.")
    required_app = {"name", "google_play_id", "apple_app_id"}
    for index, application in enumerate(data["applications"], start=1):
        missing = required_app.difference(application)
        if missing:
            raise ValueError(f"Application {index} missing keys: {sorted(missing)}")
        if not str(application["apple_app_id"]).isdigit():
            raise ValueError(f"Application {index} has an invalid Apple numeric ID.")
    return data


def utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_store_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return " ".join(text.replace("\x00", " ").split())


def stable_review_id(
    platform: str,
    application_id: str,
    raw_review_id: Any,
    review_text: str,
    review_date: datetime,
    salt: str,
) -> str:
    material = "\x1f".join(
        [
            salt,
            platform,
            application_id,
            str(raw_review_id or ""),
            review_text,
            utc_iso(review_date),
        ]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:32]


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=SCHEMA, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SCHEMA})
            count += 1
    return count


def request_json(url: str, retries: int, delay: float) -> dict[str, Any]:
    headers = {
        "User-Agent": (
            "UC-17686-Academic-Review-Collector/1.0 "
            "(public-review research; rate-limited)"
        )
    }
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            if attempt >= retries:
                raise RuntimeError(f"Request failed after {retries + 1} attempts: {url}") from exc
            wait = min(30.0, delay * (2**attempt)) + random.uniform(0.0, 0.35)
            logging.warning("Request failed (%s); retrying in %.1f seconds.", exc, wait)
            time.sleep(wait)
    raise AssertionError("unreachable")


def apple_label(node: Any, default: Any = "") -> Any:
    if isinstance(node, dict):
        return node.get("label", default)
    return default


def apple_entries(payload: dict[str, Any]) -> list[dict[str, Any]]:
    entries = payload.get("feed", {}).get("entry", [])
    if isinstance(entries, dict):
        return [entries]
    return entries if isinstance(entries, list) else []


def collect_apple(
    app: dict[str, Any],
    country: str,
    language: str,
    window: DateWindow,
    max_reviews: int,
    delay: float,
    collected_at: datetime,
    salt: str,
) -> list[dict[str, Any]]:
    app_id = str(app["apple_app_id"])
    source_url = f"https://apps.apple.com/{country}/app/id{app_id}"
    retained: list[dict[str, Any]] = []

    for page in range(1, 11):
        if len(retained) >= max_reviews:
            break
        url = (
            f"https://itunes.apple.com/{country}/rss/customerreviews/"
            f"page={page}/id={app_id}/sortby=mostrecent/json"
        )
        payload = request_json(url, retries=4, delay=delay)
        entries = apple_entries(payload)
        if not entries:
            break

        oldest_on_page: datetime | None = None
        eligible_on_page = 0
        for entry in entries:
            rating_text = apple_label(entry.get("im:rating"))
            updated_text = apple_label(entry.get("updated"))
            content = clean_text(apple_label(entry.get("content")))
            if not rating_text or not updated_text or not content:
                continue
            reviewed_at = parse_store_datetime(updated_text)
            oldest_on_page = (
                reviewed_at
                if oldest_on_page is None or reviewed_at < oldest_on_page
                else oldest_on_page
            )
            if not window.contains(reviewed_at):
                continue

            raw_review_id = apple_label(entry.get("id"))
            title = clean_text(apple_label(entry.get("title")))
            version = clean_text(apple_label(entry.get("im:version")))
            retained.append(
                {
                    "review_id": stable_review_id(
                        "ios", app_id, raw_review_id, content, reviewed_at, salt
                    ),
                    "review_text": content,
                    "review_title": title,
                    "review_rating": int(rating_text),
                    "review_datetime_utc": utc_iso(reviewed_at),
                    "application_name": app["name"],
                    "application_id": app_id,
                    "platform": "ios",
                    "store_country": country,
                    "store_language": language,
                    "app_version": version,
                    "review_likes": "",
                    "developer_response_present": False,
                    "collection_timestamp_utc": utc_iso(collected_at),
                    "source_url": source_url,
                }
            )
            eligible_on_page += 1
            if len(retained) >= max_reviews:
                break

        logging.info(
            "Apple %s page %d: %d entries, %d retained.",
            app["name"],
            page,
            len(entries),
            eligible_on_page,
        )
        if oldest_on_page and oldest_on_page < window.start:
            break
        time.sleep(delay + random.uniform(0.0, 0.25))

    return retained


def collect_google(
    app: dict[str, Any],
    country: str,
    language: str,
    window: DateWindow,
    max_reviews: int,
    delay: float,
    collected_at: datetime,
    salt: str,
) -> list[dict[str, Any]]:
    try:
        from google_play_scraper import Sort, reviews
    except ImportError as exc:
        raise RuntimeError(
            "google-play-scraper is required. Run: pip install -r requirements.txt"
        ) from exc

    app_id = app["google_play_id"]
    source_url = f"https://play.google.com/store/apps/details?id={app_id}"
    retained: list[dict[str, Any]] = []
    continuation_token = None

    while len(retained) < max_reviews:
        batch_size = min(200, max_reviews - len(retained))
        result, continuation_token = reviews(
            app_id,
            lang=language,
            country=country,
            sort=Sort.NEWEST,
            count=batch_size,
            continuation_token=continuation_token,
        )
        if not result:
            break

        oldest_in_batch: datetime | None = None
        for item in result:
            content = clean_text(item.get("content"))
            at = item.get("at")
            if not content or not at:
                continue
            reviewed_at = parse_store_datetime(at)
            oldest_in_batch = (
                reviewed_at
                if oldest_in_batch is None or reviewed_at < oldest_in_batch
                else oldest_in_batch
            )
            if not window.contains(reviewed_at):
                continue

            raw_review_id = item.get("reviewId")
            retained.append(
                {
                    "review_id": stable_review_id(
                        "android", app_id, raw_review_id, content, reviewed_at, salt
                    ),
                    "review_text": content,
                    "review_title": "",
                    "review_rating": int(item.get("score") or 0),
                    "review_datetime_utc": utc_iso(reviewed_at),
                    "application_name": app["name"],
                    "application_id": app_id,
                    "platform": "android",
                    "store_country": country,
                    "store_language": language,
                    "app_version": clean_text(
                        item.get("reviewCreatedVersion") or item.get("appVersion")
                    ),
                    "review_likes": int(item.get("thumbsUpCount") or 0),
                    "developer_response_present": bool(item.get("replyContent")),
                    "collection_timestamp_utc": utc_iso(collected_at),
                    "source_url": source_url,
                }
            )
            if len(retained) >= max_reviews:
                break

        logging.info(
            "Google %s batch: %d entries, %d retained total.",
            app["name"],
            len(result),
            len(retained),
        )
        if not continuation_token:
            break
        if oldest_in_batch and oldest_in_batch < window.start:
            break
        time.sleep(delay + random.uniform(0.0, 0.25))

    return retained


def collect_platform(
    platform: str,
    config: dict[str, Any],
    window: DateWindow,
    maximum: int,
    delay: float,
    collected_at: datetime,
    salt: str,
) -> Iterator[dict[str, Any]]:
    for application in config["applications"]:
        logging.info("Collecting %s reviews for %s.", platform, application["name"])
        if platform == "google":
            rows = collect_google(
                application,
                config["country"],
                config["language"],
                window,
                maximum,
                delay,
                collected_at,
                salt,
            )
        else:
            rows = collect_apple(
                application,
                config["country"],
                config["language"],
                window,
                maximum,
                delay,
                collected_at,
                salt,
            )
        yield from rows


def main() -> int:
    args = parse_arguments()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    config_path = Path(args.config).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(config_path)
    window = parse_window(args.date_start, args.date_end)
    collected_at = datetime.now(timezone.utc)

    selected = (
        ["google", "apple"] if args.platform == "both" else [args.platform]
    )
    manifest: dict[str, Any] = {
        "study": config["study"],
        "date_start": args.date_start,
        "date_end": args.date_end,
        "country": config["country"],
        "language": config["language"],
        "collection_timestamp_utc": utc_iso(collected_at),
        "dry_run": args.dry_run,
        "applications": config["applications"],
        "files": {},
        "limitations": [
            "Apple's public customer-review feed exposes only a subset of reviews.",
            "The Apple public feed permits at most ten pages of up to fifty reviews per app.",
            "Storefront availability and rate limits can change; all retrieval counts are reported.",
        ],
    }

    for platform in selected:
        output_path = output_dir / (
            "google_play_reviews.csv" if platform == "google" else "apple_app_store_reviews.csv"
        )
        if args.dry_run:
            count = write_csv(output_path, [])
        else:
            maximum = (
                args.max_google_per_app
                if platform == "google"
                else min(args.max_apple_per_app, 500)
            )
            count = write_csv(
                output_path,
                collect_platform(
                    platform,
                    config,
                    window,
                    maximum,
                    max(args.request_delay, 0.25),
                    collected_at,
                    args.review_id_salt,
                ),
            )
        manifest["files"][output_path.name] = {"row_count": count}

    manifest_path = output_dir / "collection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.info("Collection complete. Manifest: %s", manifest_path)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError) as error:
        logging.error("%s", error)
        raise SystemExit(2)
