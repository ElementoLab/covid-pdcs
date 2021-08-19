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

from src.types import Path, DataFrame
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
    a = sc.read(h5ad_f)
    a.obs = a.obs.replace(consts.nan_int, np.nan)

    # stratify patients based on disease timing
    t = "interval_death_symptoms_onset_days"

    a.obs.loc[a.obs[t] < 30, "timing_broad"] = "Early"
    a.obs.loc[a.obs[t] >= 30, "timing_broad"] = "Late"
    a.obs["timing_broad"] = pd.Categorical(
        a.obs["timing_broad"],
        ordered=True,
        categories=["Early", "Late"],
    )
    a.obs.loc[a.obs[t] < 10, "timing_fine"] = "Very early"
    a.obs.loc[(a.obs[t] >= 10) & (a.obs[t] < 16), "timing_fine"] = "Early"
    a.obs.loc[(a.obs[t] >= 16) & (a.obs[t] < 40), "timing_fine"] = "Middle"
    a.obs.loc[a.obs[t] >= 40, "timing_fine"] = "Late"
    a.obs["timing_fine"] = pd.Categorical(
        a.obs["timing_fine"],
        ordered=True,
        categories=["Very early", "Early", "Middle", "Late"],
    )
    return a


def load_delorey_adata():
    b = sc.read("data/scrna/lung.h5ad")
    # df = (
    #     pd.read_csv("data/scrna/lung.scp.metadata.txt", sep="\t", skiprows=0)
    #     .drop(0)
    #     .convert_dtypes()
    # )
    return b

    sc.pl.umap(b, color="SubCluster")
    # Where are the promissed pDCs from Fig 2a and ED2i???


def load_gene_signatures() -> DataFrame:
    sigs = pd.read_csv("metadata/gene_lists.csv")
    return sigs.groupby("gene_set_name")["gene_name"].apply(list).to_dict()


def cohort_structure(a: AnnData) -> None:
    meta = a.obs[consts.id_cols].drop_duplicates()
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
    marker_origin = pd.Series(
        [(g, gs) for g in markers for gs, gv in consts.dc_markers.items() if g in gv]
    )

    sel = a.obs["cell_type_intermediate"] == "Dendritic cells"
    a.obs["Dendritic cells"] = sel.astype(int)

    _a = a[a.obs.sample(frac=1).index]
    fig, axes = plt.subplots(2, 1, figsize=(4, 8))
    for ax, c in zip(axes, ["cell_type_intermediate", "Dendritic cells"]):
        sc.pl.umap(_a, color=c, ax=ax, show=False)
        ax.set_aspect(1, anchor="NW")
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "DC_location_original_UMAP.svg", **consts.figkws)
    plt.close(fig)
    del _a

    # Further cluster DCs
    a2 = a[sel, :]
    a2.obs = a2.obs.replace(consts.nan_int, np.nan)
    sc.pp.neighbors(a2)
    sc.tl.umap(a2)
    sc.tl.leiden(a2, resolution=0.4, key_added="DC_cluster")
    sc.write(consts.results_dir / "DCs.h5ad", a2)

    # # Visualize clustering
    fig = sc.pl.umap(a2, color="DC_cluster", show=False).figure
    fig.savefig(consts.results_dir / "DC_clustering.new_clusters.svg", **consts.figkws)
    plt.close(fig)

    fig = sc.pl.umap(a2, color=consts.id_cols, show=False)[0].figure
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

    # Known DC genes
    mean_exp = (
        a2.raw[:, markers].to_adata().to_df().groupby(a2.obs["DC_cluster"]).mean().T
    )
    grid1 = clustermap(mean_exp, config="abs")
    grid1.fig.savefig(
        consts.results_dir / "DC_clustering.diff_expression.DC_genes.clustermap.abs.svg",
        **consts.figkws,
    )
    plt.close(grid1.fig)
    grid2 = clustermap(resp.loc[markers], config="z")
    grid2.fig.savefig(
        consts.results_dir / "DC_clustering.diff_expression.DC_genes.clustermap.z.svg",
        **consts.figkws,
    )
    plt.close(grid2.fig)

    vmax = [max(0.1, np.percentile(a.raw[:, g].X.todense(), 99)) for g in markers]
    fig = sc.pl.umap(a2, color=markers, vmin=0, vmax=vmax, show=False)[0].figure
    fig.savefig(
        consts.results_dir / "DC_clustering.UMAP.DC_genes.svg",
        **consts.figkws,
    )
    plt.close(fig)


def score_cell_types(a):
    # Get signature scores
    sigs = load_gene_signatures()
    for sig in sigs:
        sc.tl.score_genes(a, sigs[sig], use_raw=True, score_name=sig)

    # fig, ax = plt.subplots()
    # sns.histplot(data=a2.obs[[sig]].join(a2.obs['group']), x=sig, ax=ax, label=sig, hue='group')

    signames = list(sigs.keys())
    p = a.obs.groupby(["group", "cell_type_fine"])[signames].mean()
    grid = clustermap(p)
    pdiff = p.loc["COVID-19"] - p.loc["Control"]
    grid = clustermap(pdiff, center=0, cmap="RdBu_r", metric="correlation")

    # # by timing
    _a = a[a.obs["group"] == "COVID-19"]
    p = _a.obs.groupby(["timing_broad", "cell_type_fine"])[signames].mean()
    grid = clustermap(p)

    kws = dict(center=0, cmap="RdBu_r", metric="correlation", vmin=-0.5, vmax=0.5)
    pdiff = p.loc["Early"] - p.loc["Late"]
    grid = clustermap(pdiff, **kws)

    p = _a.obs.groupby(["timing_fine", "cell_type_fine"])[signames].mean()
    grid = clustermap(p)

    kws = dict(center=0, cmap="RdBu_r", metric="correlation", vmin=-0.5, vmax=0.5)
    pdiff = p.loc["Very early"] - p.loc["Early"]
    grid = clustermap(pdiff, **kws)
    pdiff = p.loc["Early"] - p.loc["Middle"]
    grid = clustermap(pdiff, **kws)
    pdiff = p.loc["Middle"] - p.loc["Late"]
    grid = clustermap(pdiff, **kws)

    vmin = [np.percentile(a.obs[sig], 5) for sig in sigs]
    vmax = [np.percentile(a.obs[sig], 95) for sig in sigs]
    sc.pl.umap(a, color=signames, vmin=vmin, vmax=vmax)

    _a = a[a.obs["group"] == "COVID-19"]
    sc.pl.umap(_a, color=signames, vmin=vmin, vmax=vmax)
    del _a

    _a = a[(a.obs["group"] == "COVID-19") & (a.obs["immune_status"] == "Immune")]
    for sig in sigs:
        sc.pl.heatmap(
            _a,
            sigs[sig],
            groupby=["group", "cell_type_fine"],
            standard_scale=True,
            vmax=1,
        )


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
