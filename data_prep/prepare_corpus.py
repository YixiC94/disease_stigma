"""
Convert a CSV/Parquet corpus into the pickle format expected by the disease stigma
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
import pandas as pd


SENT_BOUNDARY = " SENTENCEBOUNDARYHERE "


def load_disease_mapping(disease_list_path: Path) -> dict[str, str]:
    """Load disease name variations mapping from the disease list CSV."""
    if not disease_list_path or not disease_list_path.exists():
        return {}
    
    disease_df = pd.read_csv(disease_list_path)
    disease_mapping = {}
    
    for _, row in disease_df.iterrows():
        variation = str(row['Name2']).strip().lower()
        reconciled = str(row['Reconciled_Name']).strip().lower()
        
        # Skip if either is NaN or empty
        if variation and variation != 'nan' and reconciled and reconciled != 'nan':
            disease_mapping[variation] = reconciled
    
    return disease_mapping


def consolidate_disease_names(text: str, disease_mapping: dict[str, str]) -> str:
    """Replace disease name variations with their reconciled standard names."""
    if not text or not disease_mapping:
        return text
    
    text_lower = text.lower()
    replacements = []
    
    # Sort by length (longest first) to handle overlapping patterns
    sorted_variations = sorted(disease_mapping.keys(), key=len, reverse=True)
    
    for variation in sorted_variations:
        # Use word boundaries to match whole phrases
        pattern = r'\b' + re.escape(variation) + r'\b'
        
        # Find all matches
        for match in re.finditer(pattern, text_lower):
            start, end = match.span()
            replacements.append((start, end, disease_mapping[variation]))
    
    # Sort by position (reverse) to replace from end to start (avoids index shifting)
    replacements.sort(reverse=True)
    
    # Apply replacements
    text_updated = text
    for start, end, replacement in replacements:
        text_updated = text_updated[:start] + replacement + text_updated[end:]
    
    return text_updated


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


def iter_rows(input_path: Path, *, input_format: str, encoding: str) -> Iterable[dict[str, str]]:
    fmt = input_format.lower()
    if fmt == "auto":
        suffix = input_path.suffix.lower()
        fmt = "parquet" if suffix in [".parquet", ".pq"] else "csv"

    if fmt == "parquet":
        df = pd.read_parquet(input_path)
    elif fmt == "csv":
        df = pd.read_csv(input_path, encoding=encoding)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    for record in df.to_dict(orient="records"):
        yield record


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
        # Ensure .pkl extension if not present
        if not pickle_name.endswith('.pkl'):
            pickle_name += '.pkl'
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
    parser = argparse.ArgumentParser(description="Convert a CSV/Parquet news corpus into pipeline-ready pickles.")
    parser.add_argument("--input-path", type=Path, required=True, help="Path to the input CSV or Parquet corpus.")
    parser.add_argument(
        "--input-format",
        choices=["auto", "csv", "parquet"],
        default="auto",
        help="Input format (auto-detect by default).",
    )
    parser.add_argument("--text-column", default="Text", help="Column containing the article body text.")
    parser.add_argument("--title-column", default="title", help="Optional column containing the article title.")
    parser.add_argument(
        "--date-column",
        default="Date",
        help="Column containing a date or year; used to route articles to the correct NData_<year> folder.",
    )
    parser.add_argument(
        "--disease-list-path",
        type=Path,
        default=None,
        help="Optional path to disease list CSV for consolidating disease name variations (e.g., reference_data/Disease_list_5.12.20_uncorrupted.csv).",
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
        default="corpus_{year}.pkl",
        help=(
            "Filename template for each year's pickle. Use '{year}' as a placeholder; "
            "for example 'corpus_{year}.pkl' produces clear names."
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

    # Load disease mapping if provided
    disease_mapping = {}
    if args.disease_list_path:
        disease_mapping = load_disease_mapping(args.disease_list_path)
        print(f"Loaded {len(disease_mapping)} disease name mappings from {args.disease_list_path}")

    grouped_articles: dict[int, list[str]] = defaultdict(list)
    seen_ids: set[str] = set()
    skipped_empty = 0
    skipped_short = 0
    for row in iter_rows(args.input_path, input_format=args.input_format, encoding=args.encoding):
        if args.id_column:
            row_id = clean_text(row.get(args.id_column))
            if row_id and row_id in seen_ids:
                continue
            if row_id:
                seen_ids.add(row_id)

        title = clean_text(row.get(args.title_column))
        body = clean_text(row.get(args.text_column))
        
        # Consolidate disease names if mapping is provided
        if disease_mapping:
            title = consolidate_disease_names(title, disease_mapping)
            body = consolidate_disease_names(body, disease_mapping)
        
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
        source_csv=args.input_path,
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
