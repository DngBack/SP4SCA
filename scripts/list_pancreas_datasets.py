#!/usr/bin/env python3
import argparse

import cellxgene_census
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List pancreas-related datasets from CZ CELLxGENE Census datasets table."
    )
    parser.add_argument("--keyword", default="pancreas", help="Keyword to match in dataset_title")
    parser.add_argument("--top-k", type=int, default=30, help="Number of rows to print")
    parser.add_argument("--census-version", default="stable", help="Census release tag (default: stable)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with cellxgene_census.open_soma(census_version=args.census_version) as census:
        ds = census["census_info"]["datasets"].read().concat().to_pandas()

    mask = ds["dataset_title"].str.contains(args.keyword, case=False, na=False)
    out = ds.loc[mask, ["dataset_id", "dataset_title", "dataset_total_cell_count"]]
    out = out.sort_values("dataset_total_cell_count", ascending=False).head(args.top_k)

    pd.set_option("display.max_colwidth", 200)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
