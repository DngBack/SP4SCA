#!/usr/bin/env python3
import argparse
from pathlib import Path

import cellxgene_census


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a reference/query pancreas pair from CELLxGENE by dataset_id."
    )
    parser.add_argument(
        "--ref-id",
        default="31f657dc-1875-4c4b-a5ca-ce63b3ef3a82",
        help="Dataset ID for reference h5ad",
    )
    parser.add_argument(
        "--qry-id",
        default="ff45e623-7f5f-46e3-b47d-56be0341f66b",
        help="Dataset ID for query h5ad",
    )
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--ref-name", default="pancreas_ref.h5ad", help="Reference output file name")
    parser.add_argument("--qry-name", default="pancreas_qry.h5ad", help="Query output file name")
    parser.add_argument("--census-version", default="stable", help="Census release tag (default: stable)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files if they already exist")
    return parser.parse_args()


def _download(dataset_id: str, out_path: Path, census_version: str, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        if overwrite:
            out_path.unlink()
        else:
            raise FileExistsError(f"Output already exists: {out_path}. Use --overwrite to replace it.")

    cellxgene_census.download_source_h5ad(
        dataset_id,
        to_path=str(out_path),
        census_version=census_version,
        progress_bar=True,
    )
    print(f"Downloaded {dataset_id} -> {out_path}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)

    ref_path = out_dir / args.ref_name
    qry_path = out_dir / args.qry_name

    _download(args.ref_id, ref_path, args.census_version, args.overwrite)
    _download(args.qry_id, qry_path, args.census_version, args.overwrite)

    print("\nDone. Update config.yaml if your output paths are different.")


if __name__ == "__main__":
    main()
