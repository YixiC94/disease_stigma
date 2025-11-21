"""Shared path configuration for the stigma workflow."""

from dataclasses import dataclass
from pathlib import Path
import argparse


@dataclass
class PathConfig:
    analyses_dir: Path
    raw_data_root: Path
    contemp_data_root: Path | None
    modeling_dir: Path
    results_dir: Path
    lexicon_path: Path
    disease_list_path: Path
    personality_traits_path: Path

    def raw_article_path(self, year: int) -> Path:
        return self.raw_data_root / f"NData_{year}" / f"all{year}bodytexts_regexeddisamb_listofarticles"

    def contemp_article_path(self, year: int) -> Path:
        base = self.contemp_data_root or self.raw_data_root
        return base / f"ContempData_{year}" / f"all{year}bodytexts_regexeddisamb_listofarticles"

    def bigram_path(self, start_year: int) -> Path:
        return self.modeling_dir / f"bigrammer_{start_year}_{start_year + 2}"

    def bootstrap_corpus_path(self, start_year: int) -> Path:
        return self.modeling_dir / f"allarticles_tabsep_{start_year}_{start_year + 2}tempboot"

    def bootstrap_model_path(self, start_year: int, bootnum: int, prefix: str) -> Path:
        end_year = start_year + 2
        return self.modeling_dir / "BootstrappedModels" / f"{start_year}_{end_year}" / f"{prefix}_{start_year}_{end_year}_boot{bootnum}"

    def aggregated_results_path(self, filename: str) -> Path:
        return self.results_dir / filename


def add_path_arguments(
    parser: argparse.ArgumentParser,
    *,
    require_raw_data_root: bool = True,
    require_modeling_dir: bool = True,
) -> None:
    parser.add_argument(
        "--raw-data-root",
        type=Path,
        required=require_raw_data_root,
        default=Path("data/raw"),
        help=(
            "Base directory containing NData_<year> folders with article pickles. "
            "Optional when a command does not read raw data."
        ),
    )
    parser.add_argument(
        "--contemp-data-root",
        type=Path,
        default=None,
        help="Optional base directory containing ContempData_<year> folders (falls back to raw data root).",
    )
    parser.add_argument(
        "--modeling-dir",
        type=Path,
        required=require_modeling_dir,
        default=Path("outputs/models"),
        help=(
            "Directory used for intermediate modeling artifacts (bigrams, bootstraps, and embeddings). "
            "Optional when only reading aggregated outputs."
        ),
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("outputs/results"),
        help="Directory where result CSVs and plots will be written.",
    )
    parser.add_argument(
        "--analyses-dir",
        type=Path,
        default=Path("analysis"),
        help="Base directory used for relative imports and ancillary files.",
    )
    parser.add_argument(
        "--lexicon-path",
        type=Path,
        default=Path("data/Stigma_WordLists.csv"),
        help="Path to the stigma lexicon CSV.",
    )
    parser.add_argument(
        "--disease-list-path",
        type=Path,
        default=Path("data/Disease_list_5.12.20_uncorrupted.csv"),
        help="Path to the disease list CSV.",
    )
    parser.add_argument(
        "--personality-traits-path",
        type=Path,
        default=Path("data/updated_personality_trait_list.csv"),
        help="Path to the personality traits CSV.",
    )


def build_path_config(args: argparse.Namespace) -> PathConfig:
    return PathConfig(
        analyses_dir=args.analyses_dir,
        raw_data_root=args.raw_data_root,
        contemp_data_root=args.contemp_data_root,
        modeling_dir=args.modeling_dir,
        results_dir=args.results_dir,
        lexicon_path=args.lexicon_path,
        disease_list_path=args.disease_list_path,
        personality_traits_path=args.personality_traits_path,
    )
