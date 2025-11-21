"""
Convert a CSV corpus into the pickle format expected by the disease stigma
training pipeline.

Each output file is a pickle containing a list of article strings. Sentences
within an article are separated by the sentinel string ``" SENTENCEBOUNDARYHERE "``
so downstream scripts can iterate over sentences.
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Any
import csv


SENT_BOUNDARY = " SENTENCEBOUNDARYHERE "


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    # Collapse whitespace and strip surrounding spaces/newlines.
    return re.sub(r"\s+", " ", str(value)).strip()


def split_sentences(text: str) -> list[str]:
    """Lightweight sentence segmentation.

    We avoid heavyweight dependencies by using a regex that splits on common
    sentence-ending punctuation. Empty segments are filtered out.
    """
    if not text:
        return []
    parts = re.split(r"(?<=[\.!?。！？])\s+|\n+", text)
    return [part.strip() for part in parts if part.strip()]


def extract_year(raw_value: Any, default_year: int | None) -> int:
    if isinstance(raw_value, (int, float)):
        year = int(raw_value)
        if 1900 <= year <= datetime.now().year + 1:
            return year
    if raw_value:
        # Try common date formats first.
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%d/%m/%Y"):
            try:
                return datetime.strptime(str(raw_value)[:10], fmt).year
            except ValueError:
                pass
        match = re.search(r"(?P<year>\d{4})", str(raw_value))
        if match:
            return int(match.group("year"))
    if default_year is not None:
        return default_year
    raise ValueError("Could not determine article year; provide --default-year or a parsable date column.")


def build_article_text(title: str, body: str) -> str:
    components = [segment for segment in (title, body) if segment]
    joined = " ".join(components)
    sentences = split_sentences(joined)
    return SENT_BOUNDARY.join(sentences)


def iter_rows(csv_path: Path, *, encoding: str) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding=encoding) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError("CSV appears to have no header row.")
        for row in reader:
            yield row


def write_pickles(
    grouped_articles: dict[int, list[str]],
    output_root: Path,
    *,
    basename_template: str,
    write_manifest: bool,
    manifest_name: str = "manifest.json",
    source_csv: Path,
    args: argparse.Namespace,
) -> None:
    for year, articles in grouped_articles.items():
        year_dir = output_root / f"NData_{year}"
        year_dir.mkdir(parents=True, exist_ok=True)
        pickle_name = basename_template.format(year=year)
        pickle_path = year_dir / pickle_name
        with pickle_path.open("wb") as f:
            pickle.dump(articles, f)

        if write_manifest:
            manifest_path = year_dir / manifest_name
            manifest = {
                "year": year,
                "pickle": pickle_name,
                "article_count": len(articles),
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "source_csv": str(source_csv),
                "columns": {
                    "title": args.title_column,
                    "text": args.text_column,
                    "date": args.date_column,
                    "id": args.id_column,
                },
                "default_year": args.default_year,
                "min_body_chars": args.min_body_chars,
                "processing_steps": [
                    "clean_text (collapse whitespace, strip)",
                    "split_sentences (regex-based)",
                    "join sentences with SENTENCEBOUNDARYHERE",
                    "drop duplicates by id when provided",
                    "drop empty rows",
                    "drop rows below min_body_chars",
                ],
            }
            with manifest_path.open("w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert a CSV news corpus into pipeline-ready pickles.")
    parser.add_argument("--csv-path", type=Path, required=True, help="Path to the input CSV corpus.")
    parser.add_argument("--text-column", default="Text", help="Column containing the article body text.")
    parser.add_argument("--title-column", default="title", help="Optional column containing the article title.")
    parser.add_argument(
        "--date-column",
        default="Date",
        help="Column containing a date or year; used to route articles to the correct NData_<year> folder.",
    )
    parser.add_argument(
        "--default-year",
        type=int,
        default=None,
        help="Fallback year to use when --date-column is missing or unparsable.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory to write NData_<year>/all<year>bodytexts_regexeddisamb_listofarticles pickles.",
    )
    parser.add_argument(
        "--output-basename",
        default="all{year}bodytexts_regexeddisamb_listofarticles",
        help=(
            "Filename template for each year's pickle. Use '{year}' as a placeholder; "
            "for example 'articles_{year}.pkl' produces shorter names."
        ),
    )
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="Also write a manifest.json per year describing processing steps and parameters.",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Encoding of the input CSV file (default: utf-8).",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Optional column used to drop duplicate rows (e.g., an article ID).",
    )
    parser.add_argument(
        "--min-body-chars",
        type=int,
        default=0,
        help="Skip records whose cleaned title+body length is below this threshold.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    grouped_articles: dict[int, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    skipped_empty = 0
    skipped_short = 0
    for row in iter_rows(args.csv_path, encoding=args.encoding):
        if args.id_column:
            row_id = clean_text(row.get(args.id_column))
            if row_id and row_id in seen_ids:
                continue
            if row_id:
                seen_ids.add(row_id)

        title = clean_text(row.get(args.title_column))
        body = clean_text(row.get(args.text_column))
        if not body and not title:
            skipped_empty += 1
            continue
        year = extract_year(row.get(args.date_column), args.default_year)
        article_text = build_article_text(title, body)
        if not article_text or len(article_text) < args.min_body_chars:
            skipped_short += 1
            continue
        grouped_articles[year].append(article_text)

    write_pickles(
        grouped_articles,
        args.output_root,
        basename_template=args.output_basename,
        write_manifest=args.write_manifest,
        source_csv=args.csv_path,
        args=args,
    )

    total_articles = sum(len(v) for v in grouped_articles.values())
    print(
        "Finished writing pickles:",
        f"{total_articles} kept,",
        f"{len(seen_ids)} unique IDs" if args.id_column else "IDs not used",
        f"{skipped_empty} skipped empty,",
        f"{skipped_short} skipped below length threshold",
    )


if __name__ == "__main__":
    main()
