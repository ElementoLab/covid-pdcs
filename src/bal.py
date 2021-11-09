#!/usr/bin/env python

"""
Analysis of scRNA-seq data from BALF in light of pDC action in COVID-19.
(10.1038/s41591-020-0901-9)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from anndata import AnnData
import scanpy as sc
import seaborn as sns
import pingouin as pg
from seaborn_extensions import clustermap, swarmboxenplot

from src.types import Path
from src.utils import rasterize_scanpy, load_gene_signatures


# TODO:
# # - [x] phenotype
# # - [x] quantify signatures
# # - [x] correlate signatures


def main():
    a = get_anndata()
    processing(a)
    phenotyping(a)
    score_cell_types(a)


def get_anndata(data_type: str = "raw") -> AnnData:
    import tempfile

    import requests
    import tarfile

    _dir = (consts.data_dir / "liao_nat_med_2020").mkdir()
    if data_type == "raw":
        anndata_f = _dir / "anndata.h5ad"
    elif data_type == "processed":
        anndata_f = _dir / "anndata.h5ad"
        anndata_f = anndata_f.replace_(".h5ad", ".processed.h5ad")
    elif data_type == "phenotyped":
        anndata_f = _dir / "anndata.h5ad"
        anndata_f = anndata_f.replace_(".h5ad", ".phenotyped.h5ad")

    if anndata_f.exists():
        return sc.read(anndata_f)
    if data_type != "raw":
        raise FileNotFoundError("Processed data file does not exist.")

    url = "https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE145926&format=file"
    req = requests.get(url)

    with tempfile.NamedTemporaryFile() as tmp:
        tmp.write(req.content)
    with tarfile.TarFile(tmp.name) as tar:
        tar.extractall(
            _dir, members=[m for m in tar.getmembers() if m.name.endswith(".h5")]
        )

    fs = list(Path("/home/afr/Downloads/liao_nat_med_2020").glob("*.h5"))
    anns = list()
    for f in fs:
        a = sc.read_10x_h5(f)
        a.var_names_make_unique()
        a.obs = a.obs.assign(patient=f.name.split("_")[1])
        a.obs.index = a.obs["patient"] + "-" + a.obs.index
        anns.append(a)
    ann = sc.concat(anns)

    ann.obs["disease"] = (
        ann.obs["patient"]
        .isin(["C141", "C142", "C143", "C144", "C145", "C146", "C148", "C149", "C152"])
        .replace({False: "Control", True: "COVID-19"})
    )
    ann.obs["disease"] = pd.Categorical(
        ann.obs["disease"], ordered=True, categories=["Control", "COVID-19"]
    )
    ann.obs["disease_severity"] = (
        ann.obs["patient"]
        .isin(["C141", "C142", "C144"])
        .replace({False: "Severe", True: "Mild"})
    )
    sel = ann.obs["disease"] == "Control"
    ann.obs.loc[sel, "disease_severity"] = "Control"

    ann.write(anndata_f)
    return ann


def processing(a: AnnData):
    _dir = (consts.data_dir / "liao_nat_med_2020").mkdir()
    anndata_f = _dir / "anndata.h5ad"

    a.raw = a
    sc.pp.filter_cells(a, min_counts=1000)
    sc.pp.filter_cells(a, min_genes=200)
    sc.pp.filter_cells(a, max_genes=6000)
    mito_gene_names = sc.queries.mitochondrial_genes("hsapiens").squeeze()
    a.var["mithochondrial_gene"] = a.var.index.isin(mito_gene_names)
    sc.pp.calculate_qc_metrics(
        a, qc_vars=["mithochondrial_gene"], percent_top=None, log1p=False, inplace=True
    )
    sc.pl.violin(
        a,
        ["n_genes_by_counts", "total_counts", "pct_counts_mithochondrial_gene"],
        jitter=0.4,
        multi_panel=True,
    )
    a = a[a.obs["pct_counts_mithochondrial_gene"] < 10.0]

    sc.pp.filter_genes(a, min_cells=a.shape[0] * 0.01)

    # filter based on MT/Ribosomal genes
    sc.pp.log1p(a)
    sc.pp.highly_variable_genes(a, flavor="seurat")
    a.write(anndata_f.replace_(".h5ad", ".processed_pre_regout.h5ad"))
    # a = sc.read(anndata_f.replace_(".h5ad", ".processed_pre_regout.h5ad"))

    a = a[:, a.var["highly_variable"]]
    sc.pp.regress_out(a, ["total_counts", "pct_counts_mithochondrial_gene"])
    sc.pp.scale(a, max_value=10)
    sc.pp.pca(a)

    sc.external.pp.bbknn(a, batch_key="patient")

    sc.tl.umap(a, gamma=25)
    sc.tl.leiden(a, resolution=1.0)

    a.write(anndata_f.replace_(".h5ad", ".processed.h5ad"))
    # a = sc.read(anndata_f.replace_(".h5ad", ".processed.h5ad"))


def phenotyping(a: AnnData):
    _dir = (consts.data_dir / "liao_nat_med_2020").mkdir()
    anndata_f = _dir / "anndata.h5ad"

    fig = sc.pl.umap(a, color=["leiden"], show=False).figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "clustering.leiden.svg", **consts.figkws)
    plt.close(fig)

    cell_type_label = {
        0: "Macrophages",
        1: "Macrophages",
        2: "Macrophages",
        3: "Macrophages",
        4: "Macrophages",
        5: "Macrophages",
        6: "Macrophages",
        7: "Macrophages",
        9: "Macrophages",
        13: "Macrophages",
        12: "DC/B/Plasma",
        8: "CD4 T",
        10: "CD8 T",
        14: "NK-T",
        15: "NK",
        16: "Neutrophils",
        17: "Plasma",
        11: "Epithelial",
    }

    a.obs["cell_type_label"] = a.obs["leiden"].astype(int).replace(cell_type_label)

    # Find pDCs
    markers = [y for x in consts.dc_markers.values() for y in x]
    markers += ["CD1C", "CLEC4C", "CLEC9A", "LILRA4"]
    markers = list(set(markers))

    sel = a.obs["cell_type_label"].str.contains("DC")
    dcs = a[sel, :]
    sc.tl.leiden(dcs, resolution=0.2)
    dcs.obs["cell_type_label"] = dcs.obs["leiden"].replace(
        {"0": "DC", "1": "Plasma", "2": "B", "3": "pDC", "4": "DC"}
    )
    a.obs.loc[sel, "cell_type_label"] = dcs.obs["cell_type_label"]

    a.write(anndata_f.replace_(".h5ad", ".phenotyped.h5ad"))

    a = sc.read(anndata_f.replace_(".h5ad", ".phenotyped.h5ad"))
    # Plot all cells with cell type labels
    genes = [g for g in consts.pbmc_markers if g in a.var.index.tolist()]
    vmaxes = [
        np.percentile(a.raw.X[:, a.var.index == g].todense().squeeze(), 95) for g in genes
    ]
    fig = sc.pl.umap(
        a,
        color=genes
        + ["patient", "disease", "disease_severity", "leiden", "cell_type_label"],
        use_raw=True,
        vmax=vmaxes + [None] * 4,
        show=False,
    )[0].figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "clustering.umap.high_level.svg")

    # Plot only DCs
    dcs = a[a.obs["cell_type_label"] == "pDC", :]

    markers = consts.pbmc_markers + [y for x in consts.dc_markers.values() for y in x]
    markers += ["CD1C", "CLEC4C", "CLEC9A", "LILRA4"]
    markers = list(set(markers))

    fig = sc.pl.umap(dcs, color=markers + ["leiden"], show=False)[0].figure
    lims = ((9.5, 15), (-6, 0))
    for ax in fig.axes[::2]:
        ax.set(xlim=lims[0], ylim=lims[1])
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "clustering.umap.dcs.svg")

    a.obs["cell_type_label"].to_csv(consts.results_dir / "clustering.cell_type_label.csv")

    # swarmboxenplot
    meta = (
        a.obs[["patient", "disease", "disease_severity"]]
        .drop_duplicates()
        .set_index("patient")
        .sort_index()
    )

    cell_type_count = (
        a.obs.groupby("patient")["cell_type_label"]
        .value_counts()
        .rename("count")
        .rename_axis(["patient", "cell_type_label"])
        .to_frame()
        .pivot_table(index="patient", columns="cell_type_label", values="count")
    )
    cell_type_perc = (cell_type_count.T / cell_type_count.T.sum()).T * 100

    c = dict(row_colors=meta)
    grid1 = clustermap(cell_type_count, cbar_kws=dict(label="Cell count"), **c)
    grid1.fig.savefig(
        consts.results_dir / "cell_abundance.absolute_counts.clustermap.svg"
    )
    grid2 = clustermap(
        np.log1p(cell_type_count), cbar_kws=dict(label="Cell count (log)"), **c
    )
    grid2.fig.savefig(
        consts.results_dir / "cell_abundance.absolute_counts_log.clustermap.svg"
    )
    grid3 = clustermap(cell_type_perc, cbar_kws=dict(label="% total cells"), **c)
    grid3.fig.savefig(consts.results_dir / "cell_abundance.percentage.clustermap.svg")

    fig, _ = swarmboxenplot(
        data=cell_type_perc.join(meta), x="disease", y=cell_type_perc.columns
    )
    fig.savefig(consts.results_dir / "cell_abundance.disease.swarmboxenplot.svg")

    fig, _ = swarmboxenplot(
        data=cell_type_perc.join(meta), x="disease_severity", y=cell_type_perc.columns
    )
    fig.savefig(consts.results_dir / "cell_abundance.disease_severity.swarmboxenplot.svg")

    # Clustermap of mean expression of markers
    raw = a.raw.to_adata()
    x = raw[:, markers].to_df().groupby(a.obs["cell_type_label"]).mean()

    grid = clustermap(x.T, figsize=(4, 6), cbar_kws=dict(label="Mean expression (log)"))
    grid.fig.savefig(consts.results_dir / "clustering.cell_type_means.svg")


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
    mean = _a.raw[:, ifngenes].to_adata().to_df().groupby(_a.obs["cell_type_label"]).sum()
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
    import matplotlib

    signames = list()
    for msigdb in [True, False]:
        sign = ".MSigDB" if msigdb else ""

        # Get signature scores
        sig_f = consts.results_dir / f"signature_enrichment{sign}.csv"
        if not sig_f.exists():
            sigs = load_gene_signatures(msigdb=msigdb)
            for sig in sigs:
                sc.tl.score_genes(a, sigs[sig], use_raw=True, score_name=sig)
            a.obs[sigs].to_csv(sig_f)
        else:
            sigs = load_gene_signatures(msigdb=msigdb)
            # a.obs = a.obs.join(pd.read_csv(sig_f, index_col=0))
        signames += list(sigs.keys())

    # Plot signatures at single-cell (UMAP overlay)
    signames = [
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "COVID-19 related inflammatory genes",
        "Fibrotic genes",
    ]
    vmin = 0
    [np.percentile(a.obs[sig], 0) for sig in signames]
    vmax = [np.percentile(a.obs[sig], 95) for sig in signames]
    for cmap in ["viridis", "magma", "inferno", "Reds"]:
        fig = sc.pl.umap(a, color=signames, vmin=vmin, vmax=vmax, show=False, cmap=cmap)[
            0
        ].figure
        rasterize_scanpy(fig)
        fig.savefig(
            consts.results_dir / f"signature_enrichment{sign}.UMAP.all_cells.{cmap}.svg",
            **consts.figkws,
        )

    _a = a[a.obs["disease"] == "COVID-19"]
    fig = sc.pl.umap(_a, color=signames, vmin=vmin, vmax=vmax, show=False)[0].figure
    del _a
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / f"signature_enrichment{sign}.UMAP.COVID_cells.svg",
        **consts.figkws,
    )

    cell_type_label = "cell_type_label"

    fig = sc.pl.umap(a, color=cell_type_label, show=False).figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "UMAP.cell_types.svg", **consts.figkws)

    means = a.obs.groupby(["disease", cell_type_label])[signames].mean().loc["COVID-19"]

    cts = means.index
    fig, axes = plt.subplots(
        len(signames),
        len(cts),
        figsize=(len(cts), len(signames)),
        sharey="row",
        sharex=True,
    )
    for ct, axs in zip(cts, axes.T):
        _ = swarmboxenplot(
            data=a.obs.loc[a.obs[cell_type_label] == ct],
            x="disease",
            y=signames,
            swarm=False,
            boxen=False,
            bar=True,
            ax=axs,
            test=False,
        )
        axs[0].set(title=ct)
        for ax in axs[1:]:
            ax.set(title="")
    for ax, sig in zip(axes[:, 0], signames):
        ax.set(ylabel=sig)
    fig.savefig(
        consts.results_dir / f"signature_enrichment.{cell_type_label}.barplot.svg",
        **consts.figkws,
    )

    corrs = a.obs.groupby(cell_type_label)[signames].corr()

    corr = corrs.reset_index().pivot_table(index=cell_type_label, columns="level_1")
    corr = corr.loc[:, ~(corr == 1).all()]
    corr = corr.loc[:, corr.sum().drop_duplicates().index]
    grid = clustermap(corr, cmap="RdBu_r", center=0, figsize=(7, 10), col_cluster=False)
    grid.fig.savefig(
        consts.results_dir
        / f"signature_enrichment.correlation.{cell_type_label}.heatmap.immune.svg",
        **consts.figkws,
    )

    cts = a.obs[cell_type_label].unique()
    for ct in cts:
        _a = "HALLMARK_INTERFERON_ALPHA_RESPONSE"
        _b = "COVID-19 related inflammatory genes"
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(3 * 2, 3))
        _tpc1 = a.obs.loc[
            (a.obs[cell_type_label] == ct) & (a.obs["disease"] == "Control")
        ]
        _tpc2 = a.obs.loc[
            (a.obs[cell_type_label] == ct) & (a.obs["disease"] != "Control")
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
            / f"signature_enrichment.correlation.{cell_type_label}.selected_{ct}.scatter.svg",
            **consts.figkws,
        )
        plt.close(fig)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(3 * 2, 3))
    _p = corr[(_a, _b)].sort_values()
    _r = _p.rank()
    ax.scatter(_r.index, _p)
    for i in _p.abs().sort_values().index:
        ax.text(i, _p.loc[i], s=i, ha="right")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(
        title=f"Correlation of {_a} vs\n{_b}",
        ylabel="Pearson r",
        xlabel="Cell type (rank)",
    )
    fig.savefig(
        consts.results_dir
        / f"signature_enrichment.correlation.{cell_type_label}.rank_vs_value.immune.svg",
        **consts.figkws,
    )


class consts:
    data_dir = Path("data").mkdir()
    metadata_dir = Path("metadata").mkdir()
    results_dir = (Path("results") / "balf").mkdir()
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

    pbmc_markers = [
        "TPPP3",
        "KRT18",
        "CD68",
        "FCGR3B",
        "CD1C",
        "CLEC4C",
        "CLEC9A",
        "TPSB2",
        "CD3D",
        "KLRD1",
        "MS4A1",
        "IGHG4",
    ]
    pbmc_markers = [
        # "TPPP3",
        # "KRT18",
        "CD3D",
        "CD8A",
        "CD4",
        "NKG7",
        "CD68",
        "FCGR3B",
        "CD1C",
        "CLEC4C",
        # "CLEC9A",
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
    figkws = dict(bbox_inches="tight", dpi=300)


from functools import partial

clustermap = partial(clustermap, dendrogram_ratio=0.1)


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
