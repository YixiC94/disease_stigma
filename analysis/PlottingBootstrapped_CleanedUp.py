# -*- coding: utf-8 -*-
"""
Plot bootstrapped stigma scores by disease group.
"""

import argparse
from pathlib import Path
import pandas as pd
from pylab import rcParams
import matplotlib.pyplot as plt

from config.path_config import add_path_arguments, build_path_config

rcParams["figure.figsize"] = (10, 6)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Plot bootstrapped stigma scores with confidence intervals.")
    add_path_arguments(parser, require_raw_data_root=False, require_modeling_dir=False)
    parser.add_argument("--dimension", default="stigmaindex", help="Dimension file prefix to plot.")
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Aggregated CSV to plot (defaults to results dir/<dimension>_aggregated_temp_92CI.csv).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to store generated plots (defaults to results dir).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    paths = build_path_config(args)
    input_file = args.input_file or paths.aggregated_results_path(f"{args.dimension}_aggregated_temp_92CI.csv")
    output_dir = args.output_dir or paths.results_dir

    dat = pd.read_csv(input_file)
    mycolors = [
        "Red",
        "Green",
        "Blue",
        "Yellow",
        "Orange",
        "Black",
        "Pink",
        "Purple",
        "Plum",
        "DarkRed",
        "Magenta",
        "LimeGreen",
        "Teal",
        "Cyan",
        "Goldenrod",
        "DarkGrey",
        "DarkOliveGreen",
        "Gold",
        "Wheat",
        "Peru",
        "Azure",
        "DeepPink",
        "LightCoral",
    ]

    for j in set(dat["PlottingGroup"]):
        grouped = dat[dat["PlottingGroup"] == j]
        grouped = grouped[grouped["count"] > 19]
        fig, ax = plt.subplots(1)
        diseases = sorted(set(grouped["Reconciled_Name"].values))
        for i, disease in enumerate(diseases):
            ax.plot(
                grouped[grouped["Reconciled_Name"] == disease]["Year"],
                grouped[grouped["Reconciled_Name"] == disease]["CI50%"],
                lw=2,
                label=str(disease),
                color=mycolors[i % len(mycolors)],
            )
            ax.fill_between(
                grouped[grouped["Reconciled_Name"] == disease]["Year"],
                grouped[grouped["Reconciled_Name"] == disease]["CI4%"],
                grouped[grouped["Reconciled_Name"] == disease]["CI96%"],
                facecolor=mycolors[i % len(mycolors)],
                alpha=0.5,
            )

        ax.set_title(f"{args.dimension} of diseases in plotting group: {j}")
        ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left", ncol=1)
        output_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_dir / f"{j}_{args.dimension}_Diseases_92CI_median.jpg", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
