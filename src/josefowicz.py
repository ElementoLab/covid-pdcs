#!/usr/bin/env python

"""
Analysis of scRNA-seq data from BALF in light of pDC action in COVID-19.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc
import pingouin as pg
from seaborn_extensions import clustermap, swarmboxenplot

from src.types import Path
from src.utils import rasterize_scanpy, load_gene_signatures


# TODO:
# # - [ ] XXX


def main():
    a = get_anndata()
    processing(a)


def get_anndata() -> AnnData:
    """
    These data come from the '_h5seurat_to_h5ad.py' script which converts
    data from an R object to a h5ad file
    """
    _dir = (consts.data_dir / "josefowicz").mkdir()
    anndata_f = _dir / "RNA_AllAssays_Final_Subset.scaled.h5ad"

    if anndata_f.exists():
        return sc.read(anndata_f)
    raise FileNotFoundError(
        "Could not find h5ad file. Run '_h5seurat_to_h5ad.py' script first"
    )


def cohort_structure(a: AnnData) -> None:
    a.obs["disease_severity"] = pd.Categorical(
        a.obs["Status"].str.replace("_Followup", "").replace("Other", np.nan),
        categories=["Healthy", "ICU_Control", "Mild", "Severe"],
        ordered=True,
    )
    a.obs["timepoint"] = pd.Categorical(
        a.obs["Status"]
        .str.contains("_Followup")
        .replace({False: "2-4 months", True: "4-12 months"}),
        ordered=True,
    )
    a.obs["sample_group"] = pd.Categorical(
        a.obs["Status"],
        categories=[
            "Healthy",
            "ICU_Control",
            "Mild",
            "Mild_Followup",
            "Severe",
            "Severe_Followup",
        ],
        ordered=True,
    )
    pat_meta = (
        a.obs[consts.pat_vars]
        .drop_duplicates()
        .set_index("sample")
        .sort_values(["sample_group", "disease_severity", "timepoint", "Sex", "Age"])
    )

    pat_meta.to_csv(consts.metadata_dir / "josefowicz.cohort_structure.csv")


def processing(a: AnnData) -> None:
    to_replace = {
        "CD4": "CD4+ T cell",
        "CD8": "CD8+ T cell",
        "CD14": "CD14 Monocyte",
        "CD16": "CD16 Monocyte",
        "cDC1": "Dendritic cell",
        "cDC2": "Dendritic cell",
        "ASDC": "pDC",
        "Plasmablast": "B",
    }
    to_remove = ["Doublet", "Platelet", "Eryth", "HSPC", "ILC", "MAIT", "dnT", "gdT"]
    ct = (
        a.obs["predicted.celltype.l2"]
        .str.replace("_", " ")
        .map(lambda x: x.split(" ")[0])
    )
    ct = ct.replace(to_replace)
    ct[ct.isin(to_remove)] = np.nan
    a.obs["cell_type_label"] = ct

    ct_vars = [
        "predicted.celltype.l1.score",
        "predicted.celltype.l1",
        "predicted.celltype.l2.score",
        "predicted.celltype.l2",
        "Bulkguided",
        "MarkerAnnotations",
        "ClusterAnnotations",
        "cell_type_label",
    ]

    m = [y for y in consts.pbmc_markers if y in a.raw.var.index.tolist()]
    vmaxes = [
        np.percentile(a.raw.X[:, a.raw.var.index == g].todense().squeeze(), 95) for g in m
    ]
    fig = sc.pl.umap(
        a,
        color=m
        + consts.pat_vars
        + [
            "predicted.celltype.l1",
            "predicted.celltype.l2",
            "cell_type_label",
        ],
        vmin=[0.0] * len(m) + [None] * (len(consts.pat_vars) + 3),
        vmax=vmaxes + [None] * 4,
        show=False,
    )[0].figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "phenotyping.pbmc_markers.umap.svg", **consts.figkws)
    plt.close(fig)

    # # Get expression per cell type
    # sc.tl.rank_genes_groups(a, "predicted.celltype.l2", method="t-test_overestim_var")
    # diffres = sc.get.rank_genes_groups_df(a, group=None)
    # diff_genes = (
    #     diffres.set_index("names").groupby("group")["scores"].nlargest(5).index.levels[1]
    # )

    # ct_counts = a.obs.groupby(a.obs["predicted.celltype.l2"]).size()
    # ct_means = (
    #     a.raw[:, diff_genes]
    #     .to_adata()
    #     .to_df()
    #     .groupby(a.obs["predicted.celltype.l2"])
    #     .mean()
    # )

    # grid = clustermap(
    #     ct_means,
    #     config="z",
    #     figsize=(12, 5),
    #     cbar_kws=dict(label="Expression (Z-score)"),
    #     row_colors=np.log1p(ct_counts.rename("Cell count (log)")),
    # )
    # grid.fig.savefig(
    #     consts.results_dir / "phenotyping.top_5_markers.clustermap.svg", **consts.figkws
    # )
    # plt.close(grid.fig)

    # Get cell counts per patient
    counts = (
        a.obs.groupby("sample")["predicted.celltype.l2"]
        .value_counts()
        .rename_axis(index=["sample", "predicted.celltype.l2"])
        .rename("count")
        .reset_index()
        .pivot_table(index="sample", columns="predicted.celltype.l2", values="count")
    ).drop(["Doublet"], axis=1)
    percentages = (counts.T / counts.sum(1)).T * 100
    pat_meta = (
        a.obs[consts.pat_vars]
        .drop_duplicates()
        .set_index("sample")
        .sort_values(["sample_group", "Sex", "Age"])
    )

    fig, stats = swarmboxenplot(
        data=percentages.join(pat_meta), x="sample_group", y=counts.columns
    )
    fig.savefig(
        consts.results_dir / "cell_abundance.by_severity.swarmboxenplot.svg",
        **consts.figkws,
    )
    plt.close(fig)

    counts = (
        a.obs.groupby("sample")["cell_type_label"]
        .value_counts()
        .rename_axis(index=["sample", "cell_type_label"])
        .rename("count")
        .reset_index()
        .pivot_table(index="sample", columns="cell_type_label", values="count")
    )
    percentages = (counts.T / counts.sum(1)).T * 100
    pat_meta = (
        a.obs[consts.pat_vars]
        .drop_duplicates()
        .set_index("sample")
        .sort_values(["sample_group", "Sex", "Age"])
    )

    fig, stats = swarmboxenplot(
        data=percentages.join(pat_meta), x="sample_group", y=counts.columns
    )
    fig.savefig(
        consts.results_dir
        / "cell_abundance.by_severity.swarmboxenplot.consolidated_cell_types.svg",
        **consts.figkws,
    )
    plt.close(fig)


def interferon_expression(a):
    ifngenes = a.raw.var.index[a.raw.var.index.str.startswith("IFN")]
    vmax = [max(0.1, np.percentile(a.raw[:, g].X.todense(), 99)) for g in ifngenes]
    fig = sc.pl.umap(a, color=ifngenes, vmax=vmax, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / "interferon_expression.umap.svg",
        **consts.figkws,
    )
    plt.close(fig)

    _a = a[a.obs["disease_severity"].isin(["Mild", "Severe"])]
    mean = (
        _a.raw[:, ifngenes]
        .to_adata()
        .to_df()
        .groupby(_a.obs["predicted.celltype.l2"])
        .sum()
    )
    mean = mean.loc[:, mean.var() > 0]

    grid = clustermap(
        np.log1p(mean), config="abs", cbar_kws=dict(label="Expression (log)")
    )
    grid.fig.savefig(
        consts.results_dir / "interferon_expression.mean.clustermap.abs.svg",
        **consts.figkws,
    )
    plt.close(grid.fig)
    grid = clustermap(
        mean,
        config="z",
        row_linkage=grid.dendrogram_row.linkage,
        col_linkage=grid.dendrogram_col.linkage,
        cbar_kws=dict(label="Expression (Z-score)"),
    )
    grid.fig.savefig(
        consts.results_dir / "interferon_expression.mean.clustermap.z.svg",
        **consts.figkws,
    )
    plt.close(grid.fig)


def score_cell_types(a):
    signames = list()
    for msigdb in [True, False]:
        sign = ".MSigDB" if msigdb else ""

        # Get signature scores
        sig_f = consts.results_dir / f"signature_enrichment{sign}.csv"
        if not sig_f.exists():
            sigs = load_gene_signatures(msigdb=msigdb)
            sigs = list(sigs.keys())
            for sig in sigs:
                sc.tl.score_genes(a, sigs[sig], use_raw=True, score_name=sig)
            a.obs[sigs].to_csv(sig_f)
        else:
            sigs = load_gene_signatures(msigdb=msigdb)
            sigs = list(sigs.keys())
            a.obs = a.obs.join(pd.read_csv(sig_f, index_col=0))
        signames += sigs

    # Plot signatures agregated by cell type
    ctv = "cell_type_label"

    sig_means = (
        a[
            (a.obs[ctv] != "Doublet")
            # & (a.obs["nCount_RNA"] > 1000)
            # & (a.obs["nFeature_RNA"] > 200)
            ,
            :,
        ]
        .obs.groupby(["sample", ctv, "sample_group"])[signames]
        .mean()
        .groupby(level=[ctv, "sample_group"])
        .mean()
    )

    # # replace non existing with jitter
    _m = sig_means.isnull().all(1)
    sig_means.loc[_m] = np.random.random(len(signames)) * 1e-10

    sig_diff = pd.concat(
        [
            (
                sig_means.loc[:, v, :].reset_index(level=1, drop=True)
                - sig_means.loc[:, "Healthy", :].reset_index(level=1, drop=True)
            )
            .assign(sample_group=v)
            .set_index("sample_group", append=True)
            for v in a.obs["sample_group"].cat.categories[1:]
        ]
    ).sort_index()

    grid = clustermap(
        sig_diff.T,
        col_cluster=False,
        col_colors=sig_diff.index.to_frame()[["sample_group"]],
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 9),
    )
    grid.ax_heatmap.set_xticks(range(0, sig_diff.shape[0], 5))
    grid.ax_heatmap.set_xticklabels(sig_diff.index.levels[0])
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.diff.svg",
        **consts.figkws,
    )

    grid = clustermap(
        sig_means.T,
        col_cluster=False,
        col_colors=sig_means.index.to_frame()[["sample_group"]],
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 9),
    )
    grid.ax_heatmap.set_xticks(range(0, sig_means.shape[0], 6))
    grid.ax_heatmap.set_xticklabels(sig_means.index.levels[0])
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.svg",
        **consts.figkws,
    )

    grid = clustermap(
        sig_means.T.loc[consts.soi],
        col_cluster=False,
        col_colors=sig_means.index.to_frame()[["sample_group"]],
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 3),
    )
    grid.ax_heatmap.set_xticks(range(0, sig_means.shape[0], 6))
    grid.ax_heatmap.set_xticklabels(sig_means.index.levels[0])
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.specific_sigs.svg",
        **consts.figkws,
    )

    grid = clustermap(
        sig_means.T,
        col_cluster=False,
        col_colors=sig_means.index.to_frame()[["sample_group"]],
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 9),
        z_score=0,
    )
    grid.ax_heatmap.set_xticks(range(0, sig_means.shape[0], 6))
    grid.ax_heatmap.set_xticklabels(sig_means.index.levels[0])
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.z_score.svg",
        **consts.figkws,
    )

    # Plot signatures at single-cell (UMAP overlay)
    vmin = 0
    [np.percentile(a.obs[sig], 0) for sig in consts.soi]
    vmax = [np.percentile(a.obs[sig], 95) for sig in consts.soi]
    for cmap in ["viridis", "magma", "inferno", "Reds"]:
        fig = sc.pl.umap(
            a, color=consts.soi, vmin=vmin, vmax=vmax, show=False, cmap=cmap
        )[0].figure
        rasterize_scanpy(fig)
        fig.savefig(
            consts.results_dir / f"signature_enrichment.UMAP.all_cells.{cmap}.svg",
            **consts.figkws,
        )

    _a = a[a.obs["disease"] == "COVID-19"]
    fig = sc.pl.umap(_a, color=consts.soi, vmin=vmin, vmax=vmax, show=False)[0].figure
    del _a
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / "signature_enrichment.UMAP.COVID_cells.svg",
        **consts.figkws,
    )

    # Plot as barplots
    cts = sorted(a.obs["cell_type_label"].dropna().unique())
    fig, axes = plt.subplots(
        len(consts.soi),
        len(cts),
        figsize=(len(cts), len(consts.soi)),
        sharey="row",
        sharex=True,
    )
    for ct, axs in zip(cts, axes.T):
        _ = swarmboxenplot(
            data=a.obs.loc[a.obs["cell_type_label"] == ct],
            x="sample_group",
            y=consts.soi,
            swarm=False,
            boxen=False,
            bar=True,
            ax=axs,
            plot_kws=dict(palette=consts.sample_group_colors),
            test=False,
        )
        axs[0].set(title=ct)
        for ax in axs[1:]:
            ax.set(title="")
    for ax in axes.flat:
        v = max(map(abs, ax.get_ylim()))
        ax.set_ylim((-v, v))

    for ax, sig in zip(axes[:, 0], consts.soi):
        ax.set(ylabel=sig)
    fig.savefig(
        consts.results_dir / "signature_enrichment.cell_type_label.barplot.svg",
        **consts.figkws,
    )

    diffs = pd.concat(
        (
            a.obs.query(f"cell_type_label == '{ct}' & sample_group == '{sg}'")[signames]
            - sig_means.loc[ct, "Healthy"]
        )
        .assign(cell_type_label=ct, sample_group=sg)
        .set_index(["cell_type_label", "sample_group"])
        for sg in a.obs["sample_group"].cat.categories[1:]
        for ct in cts
    ).sort_index()

    # Add a mock cell with Nan values just so that every cell type has equal columns
    s = diffs.iloc[-1]
    s.name = ("pDC", "Severe")
    s[signames] = np.nan
    diffs = diffs.append(s)

    fig, axes = plt.subplots(
        len(consts.soi),
        len(cts),
        figsize=(len(cts), len(consts.soi)),
        sharey="row",
        sharex=True,
    )
    for ct, axs in zip(cts, axes.T):
        _ = swarmboxenplot(
            data=diffs.loc[ct].reset_index(),
            x="sample_group",
            y=consts.soi,
            swarm=False,
            boxen=False,
            bar=True,
            ax=axs,
            plot_kws=dict(palette=consts.sample_group_colors[1:]),
            test=False,
        )
        axs[0].set(title=ct)
        for ax in axs[1:]:
            ax.set(title="")
    for ax in axes.flat:
        v = max(map(abs, ax.get_ylim()))
        ax.set_ylim((-v, v))

    for ax, sig in zip(axes[:, 0], consts.soi):
        ax.set(ylabel=sig)
    fig.savefig(
        consts.results_dir / "signature_enrichment.cell_type_label.barplot.diff.svg",
        **consts.figkws,
    )

    # Signature correlation
    corrs = (
        a[a.obs["sample_group"].isin(["Severe"])]
        .obs.groupby("cell_type_label")[consts.soi]
        .corr()
    )

    corr = corrs.reset_index().pivot_table(index="cell_type_label", columns="level_1")
    _corr = corr.loc[:, ~(corr == 1).all()]
    _corr = _corr.loc[:, _corr.sum().drop_duplicates().index]

    diff_corrs = (
        diffs.loc[:, "Severe", :].groupby("cell_type_label")[consts.soi].corr()
    ).dropna()
    diff_corr = diff_corrs.reset_index().pivot_table(
        index="cell_type_label", columns="level_1"
    )
    _diff_corr = diff_corr.loc[:, ~(diff_corr == 1).all()]
    _diff_corr = _diff_corr.loc[:, _diff_corr.sum().drop_duplicates().index]

    grid = clustermap(_corr, cmap="RdBu_r", center=0, figsize=(7, 10), col_cluster=False)
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.correlation.cell_type_label.heatmap.immune.svg",
        **consts.figkws,
    )

    grid = clustermap(
        _diff_corr, cmap="RdBu_r", center=0, figsize=(7, 10), col_cluster=False
    )
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.correlation.cell_type_label.heatmap.immune.diff.svg",
        **consts.figkws,
    )

    cts = a.obs["cell_type_label"].unique()
    _a = "HALLMARK_INTERFERON_ALPHA_RESPONSE"
    _b = "COVID-19 related inflammatory genes"
    for ct in cts:
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(3 * 2, 3))
        _tpc1 = a.obs.loc[
            (a.obs["cell_type_label"] == ct) & (a.obs["sample_group"] == "Healthy")
        ]
        _tpc2 = a.obs.loc[
            (a.obs["cell_type_label"] == ct) & (a.obs["sample_group"] == "Severe")
        ]
        if _tpc1.empty or _tpc2.empty:
            continue
        n = min(_tpc1.shape[0], _tpc2.shape[0])
        _tpc1 = _tpc1.sample(n=n)
        _tpc2 = _tpc2.sample(n=n)

        _tpc1[[_a, _b]] += 0.5
        _tpc2[[_a, _b]] += 0.5
        axes[0].loglog()
        axes[1].loglog()
        r1 = pg.corr(_tpc1[_a], _tpc1[_b]).squeeze()
        axes[0].set(title=f"r = {r1['r']:.3f}; p = {r1['p-val']:.2e}")
        r2 = pg.corr(_tpc2[_a], _tpc2[_b]).squeeze()
        axes[1].set(title=f"r = {r2['r']:.3f}; p = {r2['p-val']:.2e}")

        axes[0].scatter(
            _tpc1[_a],
            _tpc1[_b],
            alpha=0.2,
            s=5,
            color=sns.color_palette()[0],
            rasterized=True,
        )
        sns.regplot(
            x=_tpc1[_a],
            y=_tpc1[_b],
            ax=axes[0],
            scatter=False,
            color=sns.color_palette()[0],
        )
        axes[1].scatter(
            _tpc2[_a],
            _tpc2[_b],
            alpha=0.2,
            s=5,
            color=sns.color_palette()[1],
            rasterized=True,
        )
        sns.regplot(
            x=_tpc2[_a],
            y=_tpc2[_b],
            ax=axes[1],
            scatter=False,
            color=sns.color_palette()[1],
        )
        for ax in axes:
            ax.set(xlabel=_a, ylabel=_b)
        ct = ct.replace("/", "-")
        fig.savefig(
            consts.results_dir
            / f"signature_enrichment.correlation.cell_type_label.selected_{ct}.scatter.svg",
            **consts.figkws,
        )
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, figsize=(3 * 2, 3))
    _p = corr.loc[:, (_a, _b)].sort_values()
    # _r = _p.rank()
    ax.scatter(_p.index.values, _p)
    for i in _p.abs().sort_values().index:
        ax.text(i, _p.loc[i], s=i, ha="right")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set(
        title=f"Correlation of {_a} vs\n{_b}",
        ylabel="Pearson r",
        xlabel="Cell type (rank)",
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    fig.savefig(
        consts.results_dir
        / "signature_enrichment.correlation.cell_type_label.rank_vs_value.immune.svg",
        **consts.figkws,
    )


class consts:
    data_dir = Path("data").mkdir()
    metadata_dir = Path("metadata").mkdir()
    results_dir = (Path("results") / "josefowicz").mkdir()
    nan_int = -int(2 ** 32 / 2)
    pat_vars = [
        "sample",
        "Sex",
        "Age",
        # "Status",
        "sample_group",
        "disease_severity",
        "timepoint",
        "Obesity",
        "Diabetes",
        "MetabolicDisorders",
        "ImmuneDisorders",
    ]
    sample_group_colors = sns.color_palette("tab20")[:4] + sns.color_palette("tab20")[6:8]
    pbmc_markers = [
        # "TPPP3",
        # "KRT18",
        "CD3D",
        "CD8A",
        "CD4",
        "NKG7",
        "CD68",
        # "FCGR3B",
        "CD1C",
        # "CLEC4C",
        "CLEC9A",
        "TPSB2",
        "KLRD1",
        "MS4A1",
        "IGHG4",
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
    soi = [  # signatures of interest
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "COVID-19 related inflammatory genes",
        "Fibrotic genes",
    ]
    figkws = dict(bbox_inches="tight", dpi=300)


from functools import partial

clustermap = partial(clustermap, dendrogram_ratio=0.1)


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
