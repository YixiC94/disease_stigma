# -*- coding: utf-8 -*-
"""
Aggregate bootstrapped stigma scores for a single dimension across all time windows.
"""

import argparse
from pathlib import Path
import pandas as pd

from config.path_config import add_path_arguments, build_path_config

YEARS = [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Aggregate bootstrap CSVs for a dimension.")
    add_path_arguments(parser, require_raw_data_root=False, require_modeling_dir=False)
    parser.add_argument("--dimension", default="med", help="Dimension to aggregate (e.g., danger, disgust, impurity, med).")
    parser.add_argument(
        "--input-dir", type=Path, default=None, help="Directory containing temp<dimension><year>.csv files (defaults to results dir)."
    )
    return parser.parse_args()


def load_dimension_files(input_dir: Path, dimension: str) -> pd.DataFrame:
    frames = []
    for year in YEARS:
        frames.append(pd.read_csv(input_dir / f"temp{dimension}{year}.csv"))
    return pd.concat(frames, ignore_index=True)


def main():
    args = parse_arguments()
    paths = build_path_config(args)
    input_dir = args.input_dir or paths.results_dir

    diseases = load_dimension_files(input_dir, args.dimension)

    grouped = diseases.groupby(["Reconciled_Name", "Year"])
    aggregated = grouped.describe(percentiles=[0.04, 0.5, 0.96]).reset_index()
    aggregated2 = aggregated.merge(diseases, how="left", on=["Reconciled_Name", "Year"])
    aggregated2 = aggregated2.drop_duplicates(subset=["Reconciled_Name", "Year"])
    aggregated2 = aggregated2.drop(columns=aggregated2.columns[[2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 20, -2, -1]])
    aggregated2.columns = ["Reconciled_Name", "Year", "count", "mean", "std", "min", "CI4%", "CI50%", "CI96%", "max", "PlottingGroup"]
    aggregated2["Dimension"] = str(args.dimension)

    output_path = paths.aggregated_results_path(f"{args.dimension}_aggregated_temp_92CI.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregated2.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
