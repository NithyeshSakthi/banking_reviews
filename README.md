# Cross-Platform Mobile Banking Review Collector

This package collects genuine public reviews for the five UK banking applications identified in the UC-17686 proposal. It normalises Google Play and Apple App Store records into one privacy-minimised schema and never generates synthetic reviews.

## Included applications

| Application | Google Play package | Apple App Store ID |
|---|---|---:|
| Monzo | `co.uk.getmondo` | `1052238659` |
| Starling Bank | `com.starlingbank.android` | `956806430` |
| Chase UK | `com.chase.intl` | `1517121245` |
| Revolut | `com.revolut.revolut` | `932493382` |
| Barclays | `com.barclays.android.barclaysmobilebanking` | `536248734` |

The identifiers were verified against the UK storefronts in July 2026. Recheck them immediately before the final collection snapshot because store listings can change.

## Important limitations

- Google Play collection uses the third-party `google-play-scraper` package because Google does not provide a public research API for all reviews of apps that the researcher does not own.
- Apple collection uses the public iTunes Customer Reviews feed. The feed provides at most 10 pages of up to 50 reviews per application and therefore exposes no more than 500 public reviews per app.
- Store availability and rate limits can change. The generated manifest reports the actual retrieved counts.
- A dataset shortfall is reported honestly. The scripts never fill gaps with fabricated or generated reviews.
- Check the university’s ethics approval and the current platform terms before collection or redistribution.

## Installation

Use Python 3.10 or newer:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Validate without collecting

This checks application configuration and creates empty, correctly structured output files:

```bash
python collect_reviews.py \
  --config apps.json \
  --output-dir dry_run_output \
  --date-start 2025-07-29 \
  --date-end 2026-07-28 \
  --dry-run
```

On Windows PowerShell, enter the command on one line or replace each backslash with a backtick.

## Collect both platforms

The following example uses the rolling 12-month window ending 28 July 2026:

```bash
python collect_reviews.py \
  --config apps.json \
  --output-dir collected_reviews \
  --platform both \
  --date-start 2025-07-29 \
  --date-end 2026-07-28 \
  --max-google-per-app 4000 \
  --max-apple-per-app 500 \
  --request-delay 1.0
```

Outputs:

- `google_play_reviews.csv`
- `apple_app_store_reviews.csv`
- `collection_manifest.json`

No usernames, author IDs, profile images, or developer-response text are stored.

## Merge, clean and prepare the analytical corpus

```bash
python prepare_dataset.py \
  --inputs collected_reviews/google_play_reviews.csv collected_reviews/apple_app_store_reviews.csv \
  --output collected_reviews/unified_reviews.csv \
  --report collected_reviews/dataset_quality_report.json \
  --date-start 2025-07-29 \
  --date-end 2026-07-28 \
  --target-total 16000 \
  --seed 17686
```

This stage:

1. validates the shared schema;
2. removes empty and invalid records;
3. limits records to the approved date window;
4. filters likely non-English reviews;
5. removes duplicate IDs and exact same-day/app/platform review texts;
6. samples across application-platform strata when the eligible corpus exceeds the target; and
7. writes a transparent quality report.

If fewer than 16,000 eligible records exist, all eligible records are retained and `target_reached` is recorded as `false`.

## Recommended evidence preservation

Keep these items together:

- the unchanged configuration file;
- collection and quality manifests;
- scripts and dependency versions;
- the final review snapshot;
- a SHA-256 checksum of each file;
- the annotation guide and adjudication log; and
- the ethics approval applicable on the collection date.

Do not publish raw review text unless redistribution is permitted. A reproducibility repository can instead provide the collection code, schema, application identifiers, checksums, annotation structure and aggregate results.
