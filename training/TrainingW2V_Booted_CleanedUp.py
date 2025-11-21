# -*- coding: utf-8 -*-
"""
Train bootstrapped Word2Vec models for a 3-year window.
"""

import argparse
import time
import pickle
from random import seed, choices
from pathlib import Path
from gensim.models import Word2Vec
from gensim.models.phrases import Phraser
from nltk.tokenize import word_tokenize

from config.path_config import add_path_arguments, build_path_config


def write_booted_txt(paths, cyear: int, seed_no: int, output_path: Path):
    all_articles = []
    for year in [cyear, cyear + 1, cyear + 2]:
        try:
            with open(paths.raw_article_path(year), "rb") as file:
                tfile_split = pickle.load(file)
        except FileNotFoundError:
            with open(paths.contemp_article_path(year), "rb") as file:
                tfile_split = pickle.load(file)
        all_articles.extend(tfile_split)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        seed(seed_no)
        all_articles = choices(all_articles, k=len(all_articles))
        for article in all_articles:
            sentences_list = article.split(" SENTENCEBOUNDARYHERE ")
            for sent in sentences_list:
                f.write(sent)
                f.write("\n")


class SentenceIterator:
    def __init__(self, filepath: Path):
        self.filepath = filepath

    def __iter__(self):
        for line in open(self.filepath, "r", encoding="utf-8"):
            yield word_tokenize(line.rstrip("\n"))


class PhrasingIterable(object):
    def __init__(self, phrasifier, texts):
        self.phrasifier, self.texts = phrasifier, texts

    def __iter__(self):
        return iter(self.phrasifier[self.texts])


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train bootstrapped Word2Vec models.")
    add_path_arguments(parser)
    parser.add_argument("--year", type=int, default=1992, help="Start year of the 3-year window (e.g., 1992).")
    parser.add_argument("--boots", type=int, default=25, help="Number of bootstrap models to train.")
    parser.add_argument(
        "--model-prefix",
        type=str,
        default="CBOW_300d__win10_min50_iter3",
        help="Prefix used when saving bootstrapped Word2Vec models.",
    )
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=50)
    parser.add_argument("--vector-size", type=int, default=300)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--sleep", type=int, default=120, help="Seconds to pause between model training steps.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    paths = build_path_config(args)

    bigram_transformer = Phraser.load(str(paths.bigram_path(args.year)))

    for boot in range(args.boots):
        write_booted_txt(paths, args.year, boot, paths.bootstrap_corpus_path(args.year))
        sentences = SentenceIterator(paths.bootstrap_corpus_path(args.year))
        corpus = PhrasingIterable(bigram_transformer, sentences)
        time.sleep(args.sleep)
        model1 = Word2Vec(
            corpus,
            workers=args.workers,
            window=args.window,
            sg=0,
            size=args.vector_size,
            min_count=args.min_count,
            iter=args.iterations,
        )
        model1.init_sims(replace=True)
        model_path = paths.bootstrap_model_path(args.year, boot, args.model_prefix)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model1.save(str(model_path))
        time.sleep(args.sleep)


if __name__ == "__main__":
    main()
