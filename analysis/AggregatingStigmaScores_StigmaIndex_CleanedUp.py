# -*- coding: utf-8 -*-
"""
Aggregate bootstrapped stigma dimensions into a single stigma index.
"""

import argparse
from functools import reduce
from pathlib import Path
import pandas as pd

from config.path_config import add_path_arguments, build_path_config

YEARS = [1980, 1983, 1986, 1989, 1992, 1995, 1998, 2001, 2004, 2007, 2010, 2013, 2016]
DEFAULT_DIMENSIONS = ["negpostraits", "disgust", "danger", "impurity"]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Combine per-dimension bootstrap scores into a stigma index.")
    add_path_arguments(parser, require_raw_data_root=False, require_modeling_dir=False)
    parser.add_argument(
        "--dimensions",
        nargs="+",
        default=DEFAULT_DIMENSIONS,
        help="Dimensions to include when averaging the stigma index.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory containing per-dimension temp CSVs (defaults to results dir).",
    )
    return parser.parse_args()


def load_dimension(input_dir: Path, dimension: str) -> pd.DataFrame:
    frames = [pd.read_csv(input_dir / f"temp{dimension}{year}.csv") for year in YEARS]
    df = pd.concat(frames, ignore_index=True)
    df["bootno"] = df["BootNumber"].str.split("_", expand=True)[3]
    return df[["Reconciled_Name", "PlottingGroup", "Year", "bootno", dimension]].rename(
        columns={"PlottingGroup": f"PlottingGroup_{dimension}"}
    )


def main():
    args = parse_arguments()
    paths = build_path_config(args)
    input_dir = args.input_dir or paths.results_dir

    dimension_frames = [load_dimension(input_dir, dim) for dim in args.dimensions]
    merged = reduce(lambda left, right: left.merge(right, on=["Reconciled_Name", "Year", "bootno"]), dimension_frames)

    plotting_cols = [col for col in merged.columns if col.startswith("PlottingGroup_")]
    merged["PlottingGroup"] = merged[plotting_cols].bfill(axis=1).iloc[:, 0]
    merged = merged.drop(columns=plotting_cols)

    merged["stigma_index_mean"] = merged[args.dimensions].mean(axis=1)

    summary = (
        merged.groupby(["Reconciled_Name", "Year", "PlottingGroup"])["stigma_index_mean"]
        .agg(
            count="count",
            mean="mean",
            std="std",
            min="min",
            CI4=lambda s: s.quantile(0.04),
            CI50=lambda s: s.quantile(0.5),
            CI96=lambda s: s.quantile(0.96),
            max="max",
        )
        .reset_index()
    )
    summary["Dimension"] = "stigmaindex"

    output_path = paths.aggregated_results_path("stigmaindex_aggregated_temp_92CI.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
