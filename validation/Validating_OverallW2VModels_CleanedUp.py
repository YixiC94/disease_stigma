# -*- coding: utf-8 -*-
"""
Validate a trained Word2Vec model using WordSim-353 and Google analogy benchmarks.
"""

import argparse
from pathlib import Path
from gensim.models import Word2Vec
from gensim.test.utils import datapath


def parse_arguments():
    parser = argparse.ArgumentParser(description="Validate Word2Vec models.")
    parser.add_argument("--model-path", type=Path, required=True, help="Path to the Word2Vec model to evaluate.")
    parser.add_argument(
        "--analogy-file",
        type=Path,
        default=Path("data/questions_words_pasted.txt"),
        help="Path to the Google analogy question file.",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    model1 = Word2Vec.load(str(args.model_path))

    model1.wv.evaluate_word_pairs(datapath("wordsim353.tsv"))
    acc1, acc2 = model1.wv.evaluate_word_analogies(str(args.analogy_file))
    acc2_labels = [
        "capital-common-countries",
        "capital-world",
        "money",
        "US_capitals",
        "family",
        "adj_to_adverbs",
        "opposites",
        "comparative",
        "superlative",
        "present_participle",
        "nationality",
        "past_tense",
        "plural",
        "plural_verbs",
        "total accuracy",
    ]

    accuracy_tracker = []
    for i in range(0, len(acc2)):
        sum_corr = len(acc2[i]["correct"])
        sum_incorr = len(acc2[i]["incorrect"])
        total = sum_corr + sum_incorr
        print("Accuracy on " + str(acc2_labels[i]) + ": " + str(float(sum_corr) / (total)))
        accuracy_tracker.append(float(sum_corr) / (total))


if __name__ == "__main__":
    main()
