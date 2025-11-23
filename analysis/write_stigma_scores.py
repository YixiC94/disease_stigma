# -*- coding: utf-8 -*-
"""
Compute stigma scores for diseases across multiple dimensions and bootstrapped models.
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import normalize

from data_prep import build_lexicon_stigma
from analysis import dimension_stigma
from config.path_config import add_path_arguments, build_path_config


DEFAULT_BOOTS = 25
_PREFIX_LOG_CACHE: set[tuple[int, int, str]] = set()
_FOLDED_WARN_CACHE: set[str] = set()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Write stigma scores for each dimension and time window.")
    add_path_arguments(parser, require_raw_data_root=False)
    parser.add_argument(
        "--model-prefix",
        type=str,
        default=None,
        help=(
            "Prefix used when loading bootstrapped Word2Vec/KeyedVector models. "
            "If omitted, will attempt to read training_manifest.json under BootstrappedModels/<years>/."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for temporary per-dimension CSVs (defaults to --results-dir).",
    )
    parser.add_argument(
        "--lexicon-min-count",
        type=int,
        default=50,
        help="Minimum token count for lexicon words to be included in dimension building (default: 50).",
    )
    parser.add_argument(
        "--boots",
        type=int,
        default=DEFAULT_BOOTS,
        help="Number of bootstrapped models to load per year (default: 25).",
    )
    parser.add_argument("--start-year", type=int, default=1980, help="Start year of the window (e.g., 1992).")
    parser.add_argument(
        "--year-interval", type=int, default=3, help="Number of years to include in each window (e.g., 3 for 1992-1994)."
    )
    parser.add_argument(
        "--end-year",
        type=int,
        default=None,
        help="Optional end year; when set, will score windows from start-year to end-year stepping by year-interval.",
    )
    parser.add_argument(
        "--plotting-groups",
        nargs="+",
        default=None,
        help="Optional list of PlottingGroup names to include (e.g., neurodevelopmental). If omitted, use all groups with Plot == Yes.",
    )
    return parser.parse_args()


def fold_word(target, second, kv):
    if target not in kv.key_to_index or second not in kv.key_to_index:
        return None
    count_target = kv.get_vecattr(target, "count")
    count_second = kv.get_vecattr(second, "count")
    total = count_target + count_second
    weight_target = count_target / total
    weight_second = count_second / total
    weighted_wv = (weight_target * normalize(kv[target].reshape(1, -1))) + (
        weight_second * normalize(kv[second].reshape(1, -1))
    )
    return normalize(weighted_wv)


def add_folded_terms(model):
    kv = model.wv if hasattr(model, "wv") else model

    for target, second, new_name in [
        ("epilepsy", "epileptic", "epilepsy_folded"),
        ("drug_addiction", "drug_addict", "drug_addiction_folded"),
        ("obesity", "obese", "obesity_folded"),
    ]:
        folded = fold_word(target, second, kv)
        if folded is None:
            if new_name not in _FOLDED_WARN_CACHE:
                print(f"Skipping folded term {new_name}: missing '{target}' or '{second}' in vocab.")
                _FOLDED_WARN_CACHE.add(new_name)
            continue

        kv.add_vectors([new_name], [folded.ravel()])
        kv.set_vecattr(new_name, "count", kv.get_vecattr(target, "count") + kv.get_vecattr(second, "count"))

    return model


def melt_dimension_scores(diseases, dimension_to_plot, score_columns):
    if not score_columns:
        return pd.DataFrame(columns=["Reconciled_Name", "PlottingGroup", "Year", "BootNumber", dimension_to_plot])
    return pd.melt(
        diseases[["Reconciled_Name", "PlottingGroup", "Year", *score_columns]],
        id_vars=["Reconciled_Name", "PlottingGroup", "Year"],
        var_name="BootNumber",
        value_name=dimension_to_plot,
    )


def load_diseases(paths):
    diseases = pd.read_csv(paths.disease_list_path)
    diseases = diseases[diseases["Plot"] == "Yes"]
    # If specific plotting groups were requested via CLI, filter here.
    if hasattr(paths, "plotting_groups") and paths.plotting_groups:
        diseases = diseases[diseases["PlottingGroup"].isin(paths.plotting_groups)]
    diseases = diseases.drop_duplicates(subset=["Reconciled_Name"])
    return diseases[["PlottingGroup", "Reconciled_Name"]].copy()


def resolve_model_prefix(paths, year: int, year_interval: int, provided_prefix: str | None) -> str:
    if provided_prefix:
        return provided_prefix
    end_year = year + year_interval - 1
    manifest_path = paths.modeling_dir_base / "BootstrappedModels" / f"{year}_{end_year}" / "training_manifest.json"
    if manifest_path.exists():
        import json

        data = json.loads(manifest_path.read_text())
        prefix = data.get("model_prefix")
        if prefix:
            key = (year, end_year, prefix)
            if key not in _PREFIX_LOG_CACHE:
                print(f"[INFO] Using model_prefix from manifest: {prefix} (year {year}-{end_year})")
                _PREFIX_LOG_CACHE.add(key)
            return prefix
    raise ValueError(
        f"model-prefix not provided and manifest not found/invalid at {manifest_path}. "
        "Please pass --model-prefix explicitly."
    )


def compute_dimension_scores(
    paths,
    years,
    dimension_name,
    positive_terms,
    negative_terms,
    model_prefix,
    output_dir: Path,
    lexicon_min_count: int,
    boot_range,
    year_interval: int,
):
    print(f"[INFO] Starting dimension '{dimension_name}'")
    missing_models = 0
    no_lexicon = 0
    for yr1 in years:
        try:
            diseases = load_diseases(paths)
            score_columns = []
            resolved_prefix = resolve_model_prefix(paths, yr1, year_interval, model_prefix)
        except ValueError as e:
            print(f"[WARN] Skipping year {yr1} for dimension '{dimension_name}': {e}")
            continue
        for bootnum in boot_range:
            model_path = paths.bootstrap_model_path(yr1, bootnum, resolved_prefix)
            if not model_path.exists():
                missing_models += 1
                continue
            currentmodel1 = KeyedVectors.load(str(model_path))
            add_folded_terms(currentmodel1)
            dimension_words = build_lexicon_stigma.dimension_lexicon(
                currentmodel1, positive_terms, negative_terms, min_count=lexicon_min_count
            )
            if not dimension_words.pos_train or not dimension_words.neg_train:
                no_lexicon += 1
                continue
            dimension_obj = dimension_stigma.dimension(dimension_words, "larsen")
            kv = currentmodel1.wv if hasattr(currentmodel1, "wv") else currentmodel1
            allwordssims = dimension_obj.cos_sim(list(kv.key_to_index), returnNAs=False)
            colname = f"{dimension_name}_score_stdized_{bootnum}"
            diseases[colname] = diseases["Reconciled_Name"].apply(
                lambda x: (dimension_obj.cos_sim([str(x).lower()], returnNAs=True)[0] - np.mean(allwordssims))
                / np.std(allwordssims)
            )
            score_columns.append(colname)
        diseases["Year"] = [str(yr1)] * len(diseases)
        melted = melt_dimension_scores(diseases, dimension_name, score_columns)
        if melted.empty:
            # Still useful to know, but keep it short.
            print(f"[WARN] No scores for {dimension_name} {yr1} (boot columns empty).")
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"temp{dimension_name}{yr1}.csv"
        print(f"[INFO] Writing {len(melted)} rows for {dimension_name} {yr1} to {output_path}")
        melted.to_csv(output_path)
    if missing_models:
        print(f"[WARN] {dimension_name}: skipped {missing_models} boots due to missing models.")
    if no_lexicon:
        print(f"[WARN] {dimension_name}: skipped {no_lexicon} boots with empty lexicon overlap.")
    print(f"[INFO] Finished dimension '{dimension_name}'")


def main():
    args = parse_arguments()
    paths = build_path_config(args)
    # Attach plotting_groups to paths so load_diseases can filter without changing signature.
    paths.plotting_groups = args.plotting_groups
    output_dir = args.output_dir or args.results_dir
    boot_range = range(args.boots)
    if args.end_year is not None:
        years = list(range(args.start_year, args.end_year + 1, args.year_interval))
    else:
        years = [args.start_year]

    lexicon = pd.read_csv(paths.lexicon_path)
    lexicon = lexicon[lexicon["Removed"] != "remove"]

    dangerouswords = lexicon.loc[(lexicon["WhichPole"] == "dangerous")]["Term"].str.lower().tolist()
    safewords = lexicon.loc[(lexicon["WhichPole"] == "safe")]["Term"].str.lower().tolist()

    disgustingwords = lexicon.loc[(lexicon["WhichPole"] == "disgusting")]["Term"].str.lower().tolist()
    enticingwords = lexicon.loc[(lexicon["WhichPole"] == "enticing")]["Term"].str.lower().tolist()

    moralwords = lexicon.loc[(lexicon["WhichPole"] == "moral")]["Term"].str.lower().tolist()
    immoralwords = lexicon.loc[(lexicon["WhichPole"] == "immoral")]["Term"].str.lower().tolist()

    purewords = lexicon.loc[(lexicon["WhichPole"] == "pure")]["Term"].str.lower().tolist()
    impurewords = lexicon.loc[(lexicon["WhichPole"] == "impure")]["Term"].str.lower().tolist()

    compute_dimension_scores(
        paths,
        years,
        "danger",
        dangerouswords,
        safewords,
        args.model_prefix,
        output_dir,
        args.lexicon_min_count,
        boot_range,
        args.year_interval,
    )
    compute_dimension_scores(
        paths,
        years,
        "disgust",
        disgustingwords,
        enticingwords,
        args.model_prefix,
        output_dir,
        args.lexicon_min_count,
        boot_range,
        args.year_interval,
    )
    compute_dimension_scores(
        paths,
        years,
        "immorality",
        immoralwords,
        moralwords,
        args.model_prefix,
        output_dir,
        args.lexicon_min_count,
        boot_range,
        args.year_interval,
    )
    compute_dimension_scores(
        paths,
        years,
        "impurity",
        impurewords,
        purewords,
        args.model_prefix,
        output_dir,
        args.lexicon_min_count,
        boot_range,
        args.year_interval,
    )


if __name__ == "__main__":
    main()
