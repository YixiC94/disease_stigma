# -*- coding: utf-8 -*-
"""
Train bigrammers on a sample of the data (run once per time window).
"""

import argparse
import pickle
from random import sample
from pathlib import Path
from gensim.models import phrases
from gensim.models.phrases import Phraser

from config.path_config import add_path_arguments, build_path_config


def load_articles(paths, start_year: int, year_interval: int = 3):
    sampled_articles_for_bigrammer = []
    for year in range(start_year, start_year + year_interval):
        try:
            with open(paths.raw_article_path(year), "rb") as file:
                tfile_split = pickle.load(file)
        except FileNotFoundError:
            with open(paths.contemp_article_path(year), "rb") as file:
                tfile_split = pickle.load(file)

        samp_n = round(0.75 * len(tfile_split))
        tfile_split = sample(tfile_split, samp_n)
        tfile_split = [article.split(" SENTENCEBOUNDARYHERE ") for article in tfile_split]
        for article in tfile_split:
            for sentences_list in article:
                sentences = sentences_list.split()
                sampled_articles_for_bigrammer.append(sentences)
    return sampled_articles_for_bigrammer


def train_bigrammer(sampled_articles, save_path: Path):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    bigram_transformer = phrases.Phrases(sampled_articles, min_count=50, threshold=12)
    bigram_transformer.save(str(save_path))


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train bigram models for a configurable year window.")
    add_path_arguments(parser)
    parser.add_argument("--year", type=int, default=1992, help="Start year of the window (e.g., 1992).")
    parser.add_argument("--year-interval", type=int, default=3, help="Number of years to include in the window.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    paths = build_path_config(args)

    sampled_articles_for_bigrammer = load_articles(paths, args.year, args.year_interval)
    train_bigrammer(sampled_articles_for_bigrammer, paths.bigram_path(args.year, args.year_interval))


if __name__ == "__main__":
    main()
