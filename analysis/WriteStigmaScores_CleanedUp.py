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


YEARS = [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
DEFAULT_BOOT_RANGE = range(25)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Write stigma scores for each dimension and time window.")
    add_path_arguments(parser, require_raw_data_root=False)
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="CBOW_300d__win10_min50_iter3",
        help="Prefix used when loading bootstrapped Word2Vec/KeyedVector models.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for temporary per-dimension CSVs (defaults to --results-dir).",
    )
    return parser.parse_args()


def fold_word(target, second, wvmodel):
    weight_target = wvmodel.wv.vocab[target].count / (wvmodel.wv.vocab[target].count + wvmodel.wv.vocab[second].count)
    weight_second = wvmodel.wv.vocab[second].count / (wvmodel.wv.vocab[target].count + wvmodel.wv.vocab[second].count)
    weighted_wv = (weight_target * normalize(wvmodel.wv[target].reshape(1, -1))) + (
        weight_second * normalize(wvmodel.wv[second].reshape(1, -1))
    )
    return normalize(weighted_wv)


def add_folded_terms(model):
    epilepsy_folded = fold_word("epilepsy", "epileptic", model)
    drug_addiction_folded = fold_word("drug_addiction", "drug_addict", model)
    obesity_folded = fold_word("obesity", "obese", model)

    model.wv.add("epilepsy_folded", epilepsy_folded)
    model.wv["drug_addiction_folded"] = drug_addiction_folded
    model.wv["obesity_folded"] = obesity_folded

    model.wv.vocab["epilepsy_folded"].count = model.wv.vocab["epileptic"].count + model.wv.vocab["epilepsy"].count
    model.wv.vocab["drug_addiction_folded"].count = model.wv.vocab["drug_addict"].count + model.wv.vocab["drug_addiction"].count
    model.wv.vocab["obesity_folded"].count = model.wv.vocab["obese"].count + model.wv.vocab["obesity"].count

    return model


def melt_dimension_scores(diseases, dimension_to_plot):
    score_columns = [f"{dimension_to_plot}_score_stdized_{i}" for i in DEFAULT_BOOT_RANGE]
    return pd.melt(
        diseases[["Reconciled_Name", "PlottingGroup", "Year", *score_columns]],
        id_vars=["Reconciled_Name", "PlottingGroup", "Year"],
        var_name="BootNumber",
        value_name=dimension_to_plot,
    )


def load_diseases(paths):
    diseases = pd.read_csv(paths.disease_list_path)
    diseases = diseases[diseases["Plot"] == "Yes"]
    diseases = diseases.drop_duplicates(subset=["Reconciled_Name"])
    return diseases[["PlottingGroup", "Reconciled_Name"]].copy()


def compute_dimension_scores(paths, years, dimension_name, positive_terms, negative_terms, model_prefix, output_dir: Path):
    for yr1 in years:
        diseases = load_diseases(paths)
        for bootnum in DEFAULT_BOOT_RANGE:
            model_path = paths.bootstrap_model_path(yr1, bootnum, model_prefix)
            currentmodel1 = KeyedVectors.load(str(model_path))
            add_folded_terms(currentmodel1)
            dimension_words = build_lexicon_stigma.dimension_lexicon(currentmodel1, positive_terms, negative_terms)
            dimension_obj = dimension_stigma.dimension(dimension_words, "larsen")
            allwordssims = dimension_obj.cos_sim(list(currentmodel1.wv.vocab), returnNAs=False)
            diseases[f"{dimension_name}_score_stdized_{bootnum}"] = diseases["Reconciled_Name"].apply(
                lambda x: (dimension_obj.cos_sim([str(x).lower()], returnNAs=True)[0] - np.mean(allwordssims))
                / np.std(allwordssims)
            )
        diseases["Year"] = [str(yr1)] * len(diseases)
        melted = melt_dimension_scores(diseases, dimension_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        melted.to_csv(output_dir / f"temp{dimension_name}{yr1}.csv")


def main():
    args = parse_arguments()
    paths = build_path_config(args)
    output_dir = args.output_dir or args.results_dir

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

    compute_dimension_scores(paths, YEARS, "danger", dangerouswords, safewords, args.model_prefix, output_dir)
    compute_dimension_scores(paths, YEARS, "disgust", disgustingwords, enticingwords, args.model_prefix, output_dir)
    compute_dimension_scores(paths, YEARS, "immorality", immoralwords, moralwords, args.model_prefix, output_dir)
    compute_dimension_scores(paths, YEARS, "impurity", impurewords, purewords, args.model_prefix, output_dir)


if __name__ == "__main__":
    main()
