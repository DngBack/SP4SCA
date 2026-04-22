#!/usr/bin/env python3
import argparse
from pathlib import Path

import cellxgene_census


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download source .h5ad by CELLxGENE dataset_id.")
    parser.add_argument("--dataset-id", required=True, help="CELLxGENE dataset_id")
    parser.add_argument("--out", required=True, help="Output .h5ad path (must not already exist)")
    parser.add_argument("--census-version", default="stable", help="Census release tag (default: stable)")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and args.overwrite:
        out_path.unlink()

    cellxgene_census.download_source_h5ad(
        args.dataset_id,
        to_path=str(out_path),
        census_version=args.census_version,
        progress_bar=True,
    )
    print(f"Downloaded {args.dataset_id} -> {out_path}")


if __name__ == "__main__":
    main()
