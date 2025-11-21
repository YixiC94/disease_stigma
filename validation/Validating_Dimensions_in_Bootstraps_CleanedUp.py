"""Validate dimension quality across bootstrapped Word2Vec models."""

import argparse
from pathlib import Path
import numpy as np
"""Cross-validate stigma dimensions across bootstrapped models."""

import argparse

import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from config.path_config import add_path_arguments, build_path_config
from data_prep import build_lexicon_stigma
from analysis import dimension_stigma

YEARS = [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
DEFAULT_MODEL_PREFIX = "CBOW_300d__win10_min50_iter3"
DEFAULT_BOOT = 0


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cross-validate stigma dimensions across bootstrapped models.")
    add_path_arguments(parser)
    parser.add_argument(
        "--model-prefix",
        default=DEFAULT_MODEL_PREFIX,
        help="Model prefix used when loading bootstrapped Word2Vec models.",
    )
    parser.add_argument("--boot", type=int, default=DEFAULT_BOOT, help="Bootstrap number to evaluate (default: 0).")
    return parser.parse_args()


def load_lexicon_terms(paths) -> dict[str, list[str]]:
    lexicon = pd.read_csv(paths.lexicon_path)
    lexicon = lexicon[lexicon["Removed"] != "remove"]

    negtraits = pd.read_csv(paths.personality_traits_path)
    negtraits["Adjective"] = negtraits["Adjective"].str.lower().str.strip()
    negtraits = negtraits.drop_duplicates(subset="Adjective")

    return {
        "danger": lexicon.loc[(lexicon["WhichPole"] == "dangerous")]["Term"].str.lower().tolist(),
        "safe": lexicon.loc[(lexicon["WhichPole"] == "safe")]["Term"].str.lower().tolist(),
        "disgust": lexicon.loc[(lexicon["WhichPole"] == "disgusting")]["Term"].str.lower().tolist(),
        "entice": lexicon.loc[(lexicon["WhichPole"] == "enticing")]["Term"].str.lower().tolist(),
        "pure": lexicon.loc[(lexicon["WhichPole"] == "pure")]["Term"].str.lower().tolist(),
        "impure": lexicon.loc[(lexicon["WhichPole"] == "impure")]["Term"].str.lower().tolist(),
        "neg": negtraits[negtraits["Sentiment"] == "neg"]["Adjective"].tolist(),
        "pos": negtraits[negtraits["Sentiment"] == "pos"]["Adjective"].tolist(),
    }


def fold_dimension_vectors(model: Word2Vec, terms: dict[str, list[str]]):
    dangerwords = build_lexicon_stigma.dimension_lexicon(model, terms["danger"], terms["safe"])
    disgustwords = build_lexicon_stigma.dimension_lexicon(model, terms["disgust"], terms["entice"])
    puritywords = build_lexicon_stigma.dimension_lexicon(model, terms["impure"], terms["pure"])
    negposwords = build_lexicon_stigma.dimension_lexicon(model, terms["neg"], terms["pos"])

    return {
        "danger": dimension_stigma.dimension(dangerwords, "larsen"),
        "disgust": dimension_stigma.dimension(disgustwords, "larsen"),
        "purity": dimension_stigma.dimension(puritywords, "larsen"),
        "negpos": dimension_stigma.dimension(negposwords, "larsen"),
    }


def main() -> None:
    args = parse_arguments()
    paths = build_path_config(args)
    lexicon_terms = load_lexicon_terms(paths)

    train_accuracy_N_danger = []
    train_accuracy_percent_danger = []
    holdout_accuracy_N_danger = []
    holdout_accuracy_percent_danger = []

    train_accuracy_N_disgust = []
    train_accuracy_percent_disgust = []
    holdout_accuracy_N_disgust = []
    holdout_accuracy_percent_disgust = []

    train_accuracy_N_purity = []
    train_accuracy_percent_purity = []
    holdout_accuracy_N_purity = []
    holdout_accuracy_percent_purity = []

    train_accuracy_N_negpos = []
    train_accuracy_percent_negpos = []
    holdout_accuracy_N_negpos = []
    holdout_accuracy_percent_negpos = []

    cossim_pure_danger = []
    cossim_pure_disgust = []
    cossim_pure_negpos = []
    cossim_danger_disgust = []
    cossim_negpos_disgust = []
    cossim_negpos_danger = []

    mostsim_danger = []
    leastsim_danger = []
    mostsim_purity = []
    leastsim_purity = []
    mostsim_negpos = []
    leastsim_negpos = []
    mostsim_disgust = []
    leastsim_disgust = []

    for yr1 in YEARS:
        model_path = paths.bootstrap_model_path(yr1, args.boot, args.model_prefix)
        print(f"PROCESSING MODEL FOR YEAR: {yr1} ({model_path})")
        currentmodel = Word2Vec.load(str(model_path))

        dimensions = fold_dimension_vectors(currentmodel, lexicon_terms)

        dimtemp = dimension_stigma.kfold_dim(dimensions["disgust"].semantic_direction)
        train_accuracy_N_disgust.append(dimtemp[0])
        train_accuracy_percent_disgust.append(dimtemp[1])
        holdout_accuracy_N_disgust.append(dimtemp[2])
        holdout_accuracy_percent_disgust.append(dimtemp[3])

        dimtemp = dimension_stigma.kfold_dim(dimensions["purity"].semantic_direction)
        train_accuracy_N_purity.append(dimtemp[0])
        train_accuracy_percent_purity.append(dimtemp[1])
        holdout_accuracy_N_purity.append(dimtemp[2])
        holdout_accuracy_percent_purity.append(dimtemp[3])

        dimtemp = dimension_stigma.kfold_dim(dimensions["danger"].semantic_direction)
        train_accuracy_N_danger.append(dimtemp[0])
        train_accuracy_percent_danger.append(dimtemp[1])
        holdout_accuracy_N_danger.append(dimtemp[2])
        holdout_accuracy_percent_danger.append(dimtemp[3])

        dimtemp = dimension_stigma.kfold_dim(dimensions["negpos"].semantic_direction)
        train_accuracy_N_negpos.append(dimtemp[0])
        train_accuracy_percent_negpos.append(dimtemp[1])
        holdout_accuracy_N_negpos.append(dimtemp[2])
        holdout_accuracy_percent_negpos.append(dimtemp[3])

        cossim_pure_danger.append(
            cosine_similarity(dimensions["purity"].dimensionvec().reshape(1, -1), dimensions["danger"].dimensionvec().reshape(1, -1))
        )
        cossim_pure_disgust.append(
            cosine_similarity(dimensions["purity"].dimensionvec().reshape(1, -1), dimensions["disgust"].dimensionvec().reshape(1, -1))
        )
        cossim_pure_negpos.append(
            cosine_similarity(dimensions["purity"].dimensionvec().reshape(1, -1), dimensions["negpos"].dimensionvec().reshape(1, -1))
        )
        cossim_danger_disgust.append(
            cosine_similarity(dimensions["danger"].dimensionvec().reshape(1, -1), dimensions["disgust"].dimensionvec().reshape(1, -1))
        )
        cossim_negpos_disgust.append(
            cosine_similarity(dimensions["negpos"].dimensionvec().reshape(1, -1), dimensions["disgust"].dimensionvec().reshape(1, -1))
        )
        cossim_negpos_danger.append(
            cosine_similarity(dimensions["negpos"].dimensionvec().reshape(1, -1), dimensions["danger"].dimensionvec().reshape(1, -1))
        )

        mostsim_danger.append(currentmodel.wv.similar_by_vector(dimensions["danger"].dimensionvec(), topn=10))
        leastsim_danger.append(currentmodel.wv.similar_by_vector(-dimensions["danger"].dimensionvec(), topn=10))
        mostsim_purity.append(currentmodel.wv.similar_by_vector(dimensions["purity"].dimensionvec(), topn=10))
        leastsim_purity.append(currentmodel.wv.similar_by_vector(-dimensions["purity"].dimensionvec(), topn=10))
        mostsim_disgust.append(currentmodel.wv.similar_by_vector(dimensions["disgust"].dimensionvec(), topn=10))
        leastsim_disgust.append(currentmodel.wv.similar_by_vector(-dimensions["disgust"].dimensionvec(), topn=10))
        mostsim_negpos.append(currentmodel.wv.similar_by_vector(dimensions["negpos"].dimensionvec(), topn=10))
        leastsim_negpos.append(currentmodel.wv.similar_by_vector(-dimensions["negpos"].dimensionvec(), topn=10))

    for similarities in [
        cossim_pure_danger,
        cossim_pure_disgust,
        cossim_pure_negpos,
        cossim_danger_disgust,
        cossim_danger_disgust,
        cossim_negpos_disgust,
        cossim_negpos_danger,
    ]:
        print(round(np.mean(similarities), 2))
        print(round(np.std(similarities), 2))
        print("next")


if __name__ == "__main__":
    main()
