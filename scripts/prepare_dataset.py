#!/usr/bin/env python3
"""Merge, validate, deduplicate, language-filter, and stratify review CSVs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from collect_reviews import SCHEMA, clean_text, parse_store_datetime, utc_iso, write_csv


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the unified ABSA review corpus.")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", default="unified_reviews.csv")
    parser.add_argument("--report", default="dataset_quality_report.json")
    parser.add_argument("--date-start", required=True)
    parser.add_argument("--date-end", required=True)
    parser.add_argument("--target-total", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=17686)
    parser.add_argument(
        "--skip-language-filter",
        action="store_true",
        help="Keep all languages. English filtering is enabled by default.",
    )
    return parser.parse_args()


def read_rows(paths: Iterable[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            missing = set(SCHEMA).difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{path} is missing columns: {sorted(missing)}")
            rows.extend(dict(row) for row in reader)
    return rows


def likely_english(text: str) -> bool:
    """Deterministic English screening with a statistical detector when available."""
    cleaned = clean_text(text)
    if not cleaned:
        return False
    letters = [character for character in cleaned if character.isalpha()]
    ascii_ratio = (
        sum(ord(character) < 128 for character in letters) / len(letters)
        if letters
        else 0.0
    )
    if len(cleaned) < 20:
        return ascii_ratio >= 0.85
    try:
        from langdetect import DetectorFactory, LangDetectException, detect

        DetectorFactory.seed = 0
        try:
            return detect(cleaned) == "en"
        except LangDetectException:
            return False
    except ImportError:
        common = re.findall(
            r"\b(the|and|is|it|to|of|for|this|that|with|app|bank|not|very|my)\b",
            cleaned.lower(),
        )
        return ascii_ratio >= 0.90 and bool(common)


def normalised_duplicate_key(row: dict[str, str]) -> str:
    text = unicodedata.normalize("NFKC", row["review_text"]).casefold()
    text = re.sub(r"\s+", " ", text).strip()
    date_part = row["review_datetime_utc"][:10]
    material = "\x1f".join(
        [row["platform"], row["application_id"], date_part, text]
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def clean_and_filter(
    rows: list[dict[str, str]],
    start: datetime,
    end: datetime,
    apply_language_filter: bool,
) -> tuple[list[dict[str, str]], Counter[str]]:
    reasons: Counter[str] = Counter()
    cleaned: list[dict[str, str]] = []
    seen_review_ids: set[str] = set()
    seen_duplicates: set[str] = set()

    for original in rows:
        row = {field: clean_text(original.get(field, "")) for field in SCHEMA}
        if not row["review_text"]:
            reasons["empty_review_text"] += 1
            continue
        try:
            reviewed_at = parse_store_datetime(row["review_datetime_utc"])
        except (ValueError, TypeError):
            reasons["invalid_review_datetime"] += 1
            continue
        if reviewed_at < start or reviewed_at > end:
            reasons["outside_date_window"] += 1
            continue
        try:
            rating = int(row["review_rating"])
        except ValueError:
            reasons["invalid_rating"] += 1
            continue
        if rating not in {1, 2, 3, 4, 5}:
            reasons["invalid_rating"] += 1
            continue
        if row["platform"] not in {"android", "ios"}:
            reasons["invalid_platform"] += 1
            continue
        if apply_language_filter and not likely_english(row["review_text"]):
            reasons["non_english"] += 1
            continue
        if row["review_id"] in seen_review_ids:
            reasons["duplicate_review_id"] += 1
            continue
        duplicate_key = normalised_duplicate_key(row)
        if duplicate_key in seen_duplicates:
            reasons["duplicate_same_app_platform_date_text"] += 1
            continue

        seen_review_ids.add(row["review_id"])
        seen_duplicates.add(duplicate_key)
        cleaned.append(row)

    return cleaned, reasons


def stratified_sample(
    rows: list[dict[str, str]], target_total: int, seed: int
) -> list[dict[str, str]]:
    if target_total <= 0 or len(rows) <= target_total:
        return rows

    random_generator = random.Random(seed)
    strata: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        strata[(row["platform"], row["application_name"])].append(row)
    for values in strata.values():
        random_generator.shuffle(values)

    selected: list[dict[str, str]] = []
    active = sorted(strata)
    positions = {key: 0 for key in active}
    while len(selected) < target_total and active:
        remaining_active = []
        for key in active:
            position = positions[key]
            values = strata[key]
            if position < len(values) and len(selected) < target_total:
                selected.append(values[position])
                positions[key] += 1
            if positions[key] < len(values):
                remaining_active.append(key)
        active = remaining_active

    selected.sort(
        key=lambda row: (
            row["platform"],
            row["application_name"],
            row["review_datetime_utc"],
            row["review_id"],
        )
    )
    return selected


def summarise(
    input_count: int,
    cleaned_rows: list[dict[str, str]],
    final_rows: list[dict[str, str]],
    exclusions: Counter[str],
    target_total: int,
) -> dict[str, Any]:
    platforms = Counter(row["platform"] for row in final_rows)
    applications = Counter(
        f'{row["platform"]}:{row["application_name"]}' for row in final_rows
    )
    ratings = Counter(row["review_rating"] for row in final_rows)
    return {
        "generated_utc": utc_iso(datetime.now(timezone.utc)),
        "input_row_count": input_count,
        "eligible_after_cleaning": len(cleaned_rows),
        "target_total": target_total,
        "final_row_count": len(final_rows),
        "target_reached": len(final_rows) >= target_total,
        "exclusions": dict(sorted(exclusions.items())),
        "platform_counts": dict(sorted(platforms.items())),
        "application_platform_counts": dict(sorted(applications.items())),
        "rating_counts": dict(sorted(ratings.items())),
        "privacy_check": {
            "author_name_column_present": False,
            "author_id_column_present": False,
        },
        "limitations": [
            "The Apple public feed provides only a subset of all App Store reviews.",
            "Short-text language identification is conservative and may exclude ambiguous reviews.",
            "A target shortfall is reported rather than filled with synthetic reviews.",
        ],
    }


def main() -> int:
    args = parse_arguments()
    start = datetime.fromisoformat(args.date_start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.date_end).replace(
        hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc
    )
    rows = read_rows([Path(value) for value in args.inputs])
    cleaned_rows, exclusions = clean_and_filter(
        rows, start, end, not args.skip_language_filter
    )
    final_rows = stratified_sample(cleaned_rows, args.target_total, args.seed)
    write_csv(Path(args.output), final_rows)
    report = summarise(
        len(rows), cleaned_rows, final_rows, exclusions, args.target_total
    )
    Path(args.report).write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValueError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(2)
