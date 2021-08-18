#!/usr/bin/env python

"""
Analysis of scRNA-seq data in light of pDC action in COVID-19.
"""

import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc
from seaborn_extensions import clustermap

from src.types import Path
from src.utils import rasterize_scanpy


def main():
    a = load_adata()
    cohort_structure(a)
    find_pDCs(a)


def load_adata() -> AnnData:
    """
    Load single cell data into AnnData object.
    """
    backup_dir = Path("/media/afr/Backup_Plus/workspace/projects/covid-imc-columbia/")
    h5ad_f = backup_dir / "data" / "scrna" / "data_lungs_all_v4.h5ad"
    return sc.read(h5ad_f)


def cohort_structure(a: AnnData) -> None:
    meta = a.obs[consts.id_cols].drop_duplicates().replace(consts.nan_int, np.nan)
    meta["patient"] = meta["patient"].str.replace("addon", "")
    meta = meta.drop_duplicates().set_index("patient")

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(meta["interval_death_symptoms_onset_days"], bins=50, ax=ax)
    fig.savefig(
        consts.results_dir / "cohort_structure.temporal.symptoms_onset_to_death.svg",
        **consts.figkws,
    )
    plt.close(fig)


def find_pDCs(a: AnnData) -> None:
    markers = sorted(set(y for x in consts.dc_markers.values() for y in x))

    sel = a.obs["cell_type_intermediate"] == "Dendritic cells"
    a.obs["Dendritic cells"] = sel.astype(int)

    fig, axes = plt.subplots(2, 1)
    for ax, c in zip(axes, ["cell_type_intermediate", "Dendritic cells"]):
        sc.pl.umap(a, color=c, ax=ax, show=False)
        ax.set_aspect(1, anchor="NW")
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "DC_location_original_UMAP.svg", **consts.figkws)
    plt.close(fig)

    # Further cluster DCs
    a2 = a[sel, :]
    a2.obs = a2.obs.replace(consts.nan_int, np.nan)
    sc.pp.neighbors(a2)
    sc.tl.umap(a2)
    sc.tl.leiden(a2, resolution=0.4, key_added="DC_cluster")

    # # Visualize clustering
    fig = sc.pl.umap(a2, color="DC_cluster", show=False).figure
    fig.savefig(consts.results_dir / "DC_clustering.new_clusters.svg", **consts.figkws)
    plt.close(fig)

    fig = sc.pl.umap(a2, color=consts.id_cols, show=False).figure
    fig.savefig(consts.results_dir / "DC_clustering.patient_vars.svg", **consts.figkws)
    plt.close(fig)

    # Differential expression between DC clusters
    raw = a2.raw.to_adata()
    sc.pp.log1p(raw)
    a2.raw = raw
    sc.tl.rank_genes_groups(
        a2,
        groupby="DC_cluster",
        key_added="DC_cluster_rank_genes_groups",
        use_raw=True,
        method="t-test_overestim_var",
    )
    _res = list()
    for cluster in a2.obs["DC_cluster"].unique():
        res = sc.get.rank_genes_groups_df(
            a2, group=cluster, key="DC_cluster_rank_genes_groups"
        )
        _res.append(res.assign(cluster=cluster))
    res = pd.concat(_res)
    res.to_csv(consts.results_dir / "DC_clustering.diff_expression.csv")

    # Top significant
    resp = res.pivot_table(index="names", columns="cluster", values="scores")
    sigs = res.query("pvals_adj < 1e-15 and logfoldchanges > 0")["names"].unique()
    grid = clustermap(resp.loc[sigs], center=0, cmap="RdBu_r", yticklabels=True)
    grid.fig.savefig(
        consts.results_dir
        / "DC_clustering.diff_expression.top_differential.clustermap.svg",
        **consts.figkws,
    )
    plt.close(grid.fig)

    # Top n per cluster
    n = 15
    sigs = res.loc[range(n), "names"].unique()
    grid = clustermap(resp.loc[sigs], center=0, cmap="RdBu_r", yticklabels=True)
    grid.fig.savefig(
        consts.results_dir
        / f"DC_clustering.diff_expression.top_{n}_per_cluster.clustermap.svg",
        **consts.figkws,
    )
    plt.close(grid.fig)

    # Known gene sets
    mean_exp = (
        a2.raw[:, markers].to_adata().to_df().groupby(a2.obs["DC_cluster"]).mean().T
    )
    grid1 = clustermap(mean_exp, config="abs")
    grid2 = clustermap(resp.loc[markers], config="z")

    vmax = [np.percentile(a.raw[:, g].X.todense(), 99) for g in markers]

    sc.pl.umap(a2, color=markers, vmin=0, vmax=vmax)


class consts:
    results_dir = Path("results").mkdir()
    nan_int = -int(2 ** 32 / 2)
    id_cols = [
        "patient",
        "group",
        "age",
        "bmi",
        "sex",
        "race",
        "intubation",
        "interval_death_symptoms_onset_days",
    ]
    dc_markers = {
        "dendritic_cells": ["PTPRC", "ITGAD"],
        "cDC1": ["IRF8", "BATF3", "ID2", "CLEC9A", "CLEC9A", "XCR1", "THBD"],
        "cDC2": ["IRF4", "RELB", "ZEB2", "KLF4", "NOTCH1"],
        "pDC": [
            "ITGAD",
            "CLEC4C",
            "IL3RA",
            "NRP1",
            "TCF4",
            "IRF8",
            "RUNX1",
            "LILRA4",
            "IRF7",
        ],
    }
    figkws = dict(bbox_inches="tight", dpi=300)


from functools import partial

clustermap = partial(clustermap, dendrogram_ratio=0.1)


if __name__ == "__main__" and "get_ipython" not in locals():
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
