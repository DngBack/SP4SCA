#!/usr/bin/env python3
import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


def _generate_counts(
    n_cells: int,
    n_genes: int,
    cell_types: np.ndarray,
    study_ids: np.ndarray,
    rng: np.random.Generator,
) -> sparse.csr_matrix:
    base = rng.gamma(shape=2.0, scale=1.0, size=n_genes)

    # Add cell-type-specific signal on disjoint marker blocks.
    ct_unique = sorted(np.unique(cell_types).tolist())
    markers_per_ct = max(20, n_genes // (len(ct_unique) * 20))

    lam = np.tile(base, (n_cells, 1))

    for i, ct in enumerate(ct_unique):
        idx = np.where(cell_types == ct)[0]
        start = (i * markers_per_ct * 3) % max(1, (n_genes - markers_per_ct))
        marker_genes = np.arange(start, start + markers_per_ct)
        lam[np.ix_(idx, marker_genes)] *= 5.0

    # Mild study effect to mimic technical shift.
    st_unique = sorted(np.unique(study_ids).tolist())
    for i, st in enumerate(st_unique):
        idx = np.where(study_ids == st)[0]
        shift_block = np.arange((i * 47) % n_genes, ((i * 47) % n_genes) + 60) % n_genes
        lam[np.ix_(idx, shift_block)] *= 1.4

    # Per-cell depth variation.
    depth = rng.lognormal(mean=0.2, sigma=0.4, size=n_cells)
    lam = lam * depth[:, None] * 0.8

    counts = rng.poisson(lam).astype(np.int32)
    return sparse.csr_matrix(counts)


def _build_adata(
    n_cells: int,
    n_genes: int,
    study_pool: list[str],
    ct_probs: list[float],
    rng: np.random.Generator,
) -> ad.AnnData:
    cell_type_names = np.array(["alpha", "beta", "delta", "acinar", "ductal", "immune"], dtype=object)
    cell_types = rng.choice(cell_type_names, size=n_cells, p=np.array(ct_probs, dtype=float))
    study_ids = rng.choice(np.array(study_pool, dtype=object), size=n_cells)

    X = _generate_counts(
        n_cells=n_cells,
        n_genes=n_genes,
        cell_types=cell_types,
        study_ids=study_ids,
        rng=rng,
    )

    mt_n = 50
    var_names = [f"MT-GENE{i}" for i in range(mt_n)] + [f"GENE{i}" for i in range(mt_n, n_genes)]

    obs = pd.DataFrame(
        {
            "cell_type_coarse": cell_types,
            "study_id": study_ids,
            "batch_id": study_ids,
        },
        index=[f"cell_{i}" for i in range(n_cells)],
    )

    var = pd.DataFrame(index=var_names)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    return adata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create demo ref/query .h5ad files for SP4SCA pipeline.")
    parser.add_argument("--out-dir", default="data", help="Output directory")
    parser.add_argument("--ref-name", default="pancreas_ref.h5ad")
    parser.add_argument("--qry-name", default="pancreas_qry.h5ad")
    parser.add_argument("--n-genes", type=int, default=3600)
    parser.add_argument("--n-ref", type=int, default=2200)
    parser.add_argument("--n-qry", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    ref = _build_adata(
        n_cells=args.n_ref,
        n_genes=args.n_genes,
        study_pool=["study_A", "study_B", "study_C"],
        ct_probs=[0.22, 0.28, 0.1, 0.16, 0.16, 0.08],
        rng=rng,
    )
    qry = _build_adata(
        n_cells=args.n_qry,
        n_genes=args.n_genes,
        study_pool=["study_D", "study_E"],
        ct_probs=[0.16, 0.32, 0.1, 0.14, 0.18, 0.1],
        rng=rng,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_path = out_dir / args.ref_name
    qry_path = out_dir / args.qry_name
    ref.write_h5ad(ref_path)
    qry.write_h5ad(qry_path)

    print(f"Wrote {ref_path} with shape {ref.shape}")
    print(f"Wrote {qry_path} with shape {qry.shape}")


if __name__ == "__main__":
    main()
