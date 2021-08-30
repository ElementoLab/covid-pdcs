#!/usr/bin/env python

"""
Analysis of scRNA-seq data in light of pDC action in COVID-19.
"""

import sys
import typing as tp

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from anndata import AnnData
import scanpy as sc
from seaborn_extensions import clustermap

from src.types import Path
from src.utils import rasterize_scanpy


# TODO:
# # - [x] find more pDCs
# # - [x] ISG/IFNa signature/expression in pDCs
# # - [x] pDC unbalance: test, get early/late
# # - [x] Correlate signatures for each cell type individually


def main():
    a = load_adata()
    cohort_structure(a)
    find_pDCs(a)
    score_cell_types(a)


def load_adata() -> AnnData:
    """
    Load single cell data into AnnData object.
    """
    backup_dir = Path("/media/afr/Backup_Plus1/workspace/projects/covid-imc-columbia/")
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


def load_gene_signatures(msigdb: bool = False) -> tp.Dict:
    if not msigdb:
        sigs = pd.read_csv("metadata/gene_lists.csv")
        return sigs.groupby("gene_set_name")["gene_name"].apply(list).to_dict()
    msigdb_f = Path("h.all.v7.4.symbols.gmt")

    msigdb_h = msigdb_f.open()
    sets = dict()
    for line in msigdb_h.readlines():
        s = line.strip().split("\t")
        sets[s[0]] = s[2:]
    return sets


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

    _a = a[a.obs["group"] == "COVID-19"]
    mean = _a.raw[:, ifngenes].to_adata().to_df().groupby(_a.obs["cell_type_fine"]).mean()
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

    pdc_file = consts.metadata_dir / "pDC_identity.txt"
    with open(pdc_file, "w") as handle:
        handle.write("\n".join(a2.obs.index[a2.obs["DC_cluster"] == "4"].tolist()))

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

    bal = (
        a2.obs.groupby("DC_cluster")["group"].value_counts()
        / a2.obs.groupby("DC_cluster")["group"].size()
    )

    import scipy
    import pingouin as pg

    c = a2.obs.groupby("DC_cluster")["group"].value_counts()
    _stats = list()
    for cluster in range(5):
        _a = c.loc[str(cluster), "COVID-19"]
        _b = c.loc[str(cluster), "Control"]
        _c = c.loc[c.index.get_level_values(0) != str(cluster), "COVID-19"].sum()
        _d = c.loc[c.index.get_level_values(0) != str(cluster), "Control"].sum()
        s, p = scipy.stats.fisher_exact([[_a, _b], [_c, _d]])
        _stats.append([cluster, _a, _b, _c, _d, s, p])
    stats = pd.DataFrame(_stats, columns=["cluster", "a", "b", "c", "d", "stat", "p-unc"])
    stats["p-cor"] = pg.multicomp(stats["p-unc"].values)[1]
    stats.to_csv(consts.results_dir / "DC_clustering.cluster_unbalance.stats.csv")

    clusters = a2.obs["DC_cluster"].unique()
    fig, axes = plt.subplots(1, len(clusters), sharey=True)
    for ax, c in zip(axes, clusters):
        sns.barplot(x=bal.loc[c].index, y=bal.loc[c], ax=ax)
    fig.savefig(
        consts.results_dir / "DC_clustering.cluster_unbalance.svg",
        **consts.figkws,
    )
    plt.close(fig)

    _a2 = a2[a2.obs["group"] == "COVID-19"]
    ax = sc.pl.violin(
        _a2, groupby="DC_cluster", keys="interval_death_symptoms_onset_days", show=False
    )

    q = a2.obs.groupby("DC_cluster")["interval_death_symptoms_onset_days"].mean()
    for c, v in q.iteritems():
        ax.axhline(v, color=sns.color_palette()[int(c)])

    ax.figure.savefig(
        consts.results_dir / "DC_clustering.cluster_unbalance-time_to_death.svg",
        **consts.figkws,
    )
    plt.close(fig)
    del _a2

    # from seaborn_extensions import swarmboxenplot
    # fig, stats = swarmboxenplot(data=a2.obs, x='DC_cluster', y='interval_death_symptoms_onset_days')

    # Measure signatures on DCs
    a2 = sc.read(consts.results_dir / "DCs.h5ad")
    sigs = load_gene_signatures(msigdb=False)
    signames = list(sigs.keys())
    for sig in sigs:
        sc.tl.score_genes(a2, sigs[sig], use_raw=True, score_name=sig)
    fig = sc.pl.umap(a2, color=signames, show=False)[0].figure
    fig.savefig(
        consts.results_dir / "DC_clustering.signature_enrichment.UMAP.svg",
        **consts.figkws,
    )
    plt.close(fig)

    fig = sc.pl.violin(a2, groupby="DC_cluster", keys=signames, show=False)[0].figure
    fig, stats = swarmboxenplot(data=a2.obs, x="DC_cluster", y=signames, hue="")
    for sig in signames:
        sign = (a2.obs[sig] > 0).astype(int).replace(0, -1)
        a2.obs[sig + "_log"] = a2.obs[sig].abs() ** (1 / 3) * sign
    signames_log = [s + "_log" for s in signames]
    fig = sc.pl.violin(a2, groupby="DC_cluster", keys=signames_log, show=False)[0].figure

    ifngenes = a.raw.var.index[a.raw.var.index.str.startswith("IFN")]
    vmax = [max(0.1, np.percentile(a.raw[:, g].X.todense(), 99)) for g in ifngenes]
    fig = sc.pl.umap(a, color=ifngenes, vmax=vmax, show=False)[0].figure


def score_cell_types(a, msigdb: bool = False):
    import matplotlib

    sign = ".MSigDB" if msigdb else ""

    # Get signature scores
    sig_f = consts.results_dir / f"signature_enrichment{sign}.csv"
    if not sig_f.exists():
        sigs = load_gene_signatures(msigdb=msigdb)
        signames = list(sigs.keys())
        for sig in sigs:
            sc.tl.score_genes(a, sigs[sig], use_raw=True, score_name=sig)
        a.obs[signames].to_csv(sig_f)
    else:
        sigs = load_gene_signatures(msigdb=msigdb)
        signames = list(sigs.keys())
        a.obs = a.obs.join(pd.read_csv(sig_f, index_col=0))

    # Plot signatures at single-cell (UMAP overlay)
    signames = [
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "COVID-19 related inflammatory genes",
        "Fibrotic genes",
    ]
    vmin = [np.percentile(a.obs[sig], 5) for sig in sigs]
    vmax = [np.percentile(a.obs[sig], 95) for sig in sigs]
    fig = sc.pl.umap(a, color=sigs, vmin=vmin, vmax=vmax, show=False)[0].figure
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / f"signature_enrichment{sign}.UMAP.all_cells.svg",
        **consts.figkws,
    )

    _a = a[a.obs["group"] == "COVID-19"]
    fig = sc.pl.umap(_a, color=signames, vmin=vmin, vmax=vmax, show=False)[0].figure
    del _a
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / f"signature_enrichment{sign}.UMAP.COVID_cells.svg",
        **consts.figkws,
    )

    cell_type_label = "cell_type_main"
    cell_type_label = "cell_type_fine"
    cell_type_label = "cell_type_intermediate"
    pdc_file = consts.metadata_dir / "pDC_identity.txt"
    pdcs = pdc_file.read_text().split("\n")
    a.obs[cell_type_label] = a.obs[cell_type_label].cat.add_categories(
        ["Plasmacytoid dendritic cells"]
    )
    a.obs.loc[pdcs, cell_type_label] = "Plasmacytoid dendritic cells"
    a.obs[cell_type_label] = a.obs[cell_type_label].cat.reorder_categories(
        a.obs[cell_type_label].cat.categories.sort_values()
    )

    fig = sc.pl.umap(a, color=cell_type_label, show=False).figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / f"UMAP.cell_types.svg", **consts.figkws)

    immune_cell_types = a.obs.loc[
        a.obs["immune_status"] == "Immune", cell_type_label
    ].drop_duplicates()
    myeloid_cell_types = immune_cell_types[
        immune_cell_types.str.contains("acrophage|onocyte|lasmacytoid|endritic")
    ]
    fig = sc.pl.umap(
        a, color=cell_type_label, show=False, groups=myeloid_cell_types.tolist()
    ).figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / f"UMAP.cell_types.myeloid.svg", **consts.figkws)

    norm = matplotlib.colors.Normalize(vmin=-0.3, vmax=0.3)
    means = a.obs.groupby(["group", cell_type_label])[signames].mean().loc["COVID-19"]

    fig, axes = plt.subplots(1, len(signames), figsize=(len(signames) * 7, 3.5))
    for ax, c in zip(axes, means):
        m = means[c].sort_values(ascending=False)
        ax.bar(m.index.tolist(), m, color=plt.get_cmap("RdBu_r")(norm(m)))
        ax.set_xticklabels(m.index.tolist(), rotation=90)
        ax.axhline(0, linestyle="--", color="grey")
        ax.set(ylabel=c)
    fig.savefig(
        consts.results_dir
        / f"signature_enrichment.rank_vs_value.{cell_type_label}.COVID-19_only.svg",
        **consts.figkws,
    )

    grid = clustermap(means[signames], config="z")
    grid = clustermap(
        means[signames], config="abs", figsize=(5, 10), metric="correlation"
    )

    p = a.obs.groupby([cell_type_label, "group"])[signames].mean()
    grid = clustermap(
        p.loc[myeloid_cell_types, signames],
        row_cluster=False,
        config="abs",
        figsize=(4, 5),
        metric="correlation",
        center=0,
        cmap="RdBu_r",
        row_colors=p.index.to_frame()["group"],
        standard_scale=True,
    )
    grid.fig.savefig(
        consts.results_dir / f"signature_enrichment.heatmap.{cell_type_label}.both.svg",
        **consts.figkws,
    )

    p = a.obs.groupby(["group", cell_type_label])[signames].mean()
    pdiff = p.loc["COVID-19"] - p.loc["Control"]
    grid = clustermap(
        pdiff[signames],
        config="abs",
        figsize=(5, 10),
        metric="correlation",
        center=0,
        cmap="RdBu_r",
    )

    # Correlate signatures per cell type
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].scatter(a.obs[signames[-1]], a.obs[signames[0]], s=2, alpha=0.2)
    axes[1].scatter(a.obs[signames[-1]], a.obs[signames[1]], s=2, alpha=0.2)

    # a.obs[signames].corr()
    corrs = a.obs.groupby(cell_type_label)[signames].corr()

    # grid = clustermap(corrs)
    corr = (
        corrs
        # .drop(signames[-1], level=1)
        .reset_index().pivot_table(index=cell_type_label, columns="level_1")
    ).loc[immune_cell_types]
    grid = clustermap(corr, cmap="RdBu_r", center=0, figsize=(7, 10), col_cluster=False)
    grid.fig.savefig(
        consts.results_dir
        / f"signature_enrichment.correlation.{cell_type_label}.heatmap.immune.svg",
        **consts.figkws,
    )

    import pingouin as pg

    cts = a.obs[cell_type_label].unique()
    # cts = ['Alveolar macrophages', 'Monocyte-derived macrophages', 'Monocytes']
    for ct in cts:
        _a = "HALLMARK_INTERFERON_ALPHA_RESPONSE"
        _b = "COVID-19 related inflammatory genes"
        fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(3 * 2, 3))
        _tpc1 = a.obs.loc[(a.obs[cell_type_label] == ct) & (a.obs["group"] == "Control")]
        _tpc2 = a.obs.loc[(a.obs[cell_type_label] == ct) & (a.obs["group"] != "Control")]
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
    _p = corr.loc[immune_cell_types][(_a, _b)]
    _r = _p.rank()
    ax.scatter(_r, _p)
    for i in _p.abs().sort_values().tail(8).index:
        ax.text(_r.loc[i], _p.loc[i], s=i, ha="right")
    ax.axhline(0, linestyle="--", color="grey")
    fig.savefig(
        consts.results_dir
        / f"signature_enrichment.correlation.{cell_type_label}.rank_vs_value.immune.svg",
        **consts.figkws,
    )

    # _a = a[(a.obs["group"] == "COVID-19") & (a.obs["immune_status"] == "Immune")]
    # for sig in sigs:
    #     sc.pl.heatmap(
    #         _a,
    #         sigs[sig],
    #         groupby=["group", cell_type_label],
    #         standard_scale=True,
    #         vmax=1,
    #     )

    # Group signatures by cell type
    oprefix = consts.results_dir / f"signature_enrichment{sign}.{cell_type_label}."

    # Observe signature enrichment per cell type
    a.obs["group"] = a.obs["group"].cat.reorder_categories(
        ["Control", "COVID-19"], ordered=True
    )
    p = a.obs.groupby([cell_type_label, "group"])[signames].mean()
    grid = clustermap(
        p,
        row_cluster=False,
        cmap="RdBu_r",
        center=0,
        vmin=-0.2,
        vmax=0.2,
        cbar_kws=dict(label="Signature enrichment"),
    )
    grid.savefig(oprefix + "svg", **consts.figkws)

    _p = a.obs.groupby(["group", cell_type_label])[signames].mean()
    pdiff = _p.loc["COVID-19"] - _p.loc["Control"]
    paths = pdiff.mean().sort_values().tail(15).index

    grid = clustermap(
        p[paths],
        row_cluster=False,
        cmap="RdBu_r",
        center=0,
        vmin=-0.2,
        vmax=0.2,
        cbar_kws=dict(label="Signature enrichment"),
        figsize=(5, 8),
        row_colors=p.index.to_frame()[["group"]],
    )
    grid.savefig(oprefix + "selected_variable.svg", **consts.figkws)

    grid = clustermap(
        p[paths],
        row_cluster=False,
        cmap="RdBu_r",
        center=0,
        # vmin=-0.2, vmax=0.2,
        cbar_kws=dict(label="Signature enrichment"),
        figsize=(5, 8),
        row_colors=p.index.to_frame()[["group"]],
        standard_scale=1,
    )
    grid.savefig(oprefix + "selected_variable.scaled.svg", **consts.figkws)

    grid = clustermap(
        p.loc[immune_cell_types, paths].sort_index(),
        row_cluster=False,
        cmap="RdBu_r",
        center=0,
        vmin=-0.2,
        vmax=0.2,
        cbar_kws=dict(label="Signature enrichment"),
        figsize=(5, 7),
        row_colors=p.index.to_frame()[["group"]],
    )
    grid.savefig(oprefix + "selected_variable.immune_only.svg", **consts.figkws)

    grid = clustermap(
        p.loc[immune_cell_types, paths].sort_index(),
        row_cluster=False,
        cmap="RdBu_r",
        center=0,
        # vmin=-0.2, vmax=0.2,
        cbar_kws=dict(label="Signature enrichment"),
        figsize=(5, 7),
        row_colors=p.index.to_frame()[["group"]],
        standard_scale=1,
    )
    grid.savefig(oprefix + "selected_variable.immune_only.scaled.svg", **consts.figkws)

    grid = clustermap(p, row_cluster=False, config="z")
    grid.savefig(oprefix + "z.svg", **consts.figkws)

    p = a.obs.groupby(["group", cell_type_label])[signames].mean()
    for group in ["Control", "COVID-19"]:
        grid = clustermap(p.loc[group])
        grid.savefig(oprefix + f"per_disease_{group}.svg", **consts.figkws)

        from imc.graphics import get_grid_dims

        fig = get_grid_dims(signames, return_fig=True)
        for i, sig in enumerate(signames):
            sns.barplot(
                x=p.loc[group, sig],
                y=p.loc[group, sig].index,
                ax=fig.axes[i],
                orient="horiz",
            )
        fig.tight_layout()
        fig.savefig(oprefix + f"per_disease_{group}.barplot.svg", **consts.figkws)

    # fig, axes = plt.subplots(1, 3)
    # for i, sig in enumerate(signames):
    #     for j, group in enumerate(["Control", "COVID-19"]):
    #         sns.histplot(
    #             a.obs.query(f"group == '{group}'")[sig],
    #             ax=axes[i],
    #             label=group,
    #             color=sns.color_palette()[j],
    #             stat='percent'
    #         )

    fig, axes = plt.subplots(1, 3, figsize=(3 * 4, 4))
    for i, sig in enumerate(signames):
        for j, group in enumerate(["Control", "COVID-19"]):
            sns.histplot(
                p.loc[group, sig],
                ax=axes[i],
                label=group,
                color=sns.color_palette()[j],
                stat="percent",
            )
        axes[i].set_title(sig)
    fig.savefig(oprefix + "per_disease.grouped_cell_type.histplot.svg", **consts.figkws)

    p = a.obs.groupby(["group", cell_type_label])[signames].mean()
    pdiff = p.loc["COVID-19"] - p.loc["Control"]
    pz = ((p.T - p.mean(1)) / p.std(1)).T
    pdiff = pz.loc["COVID-19"] - pz.loc["Control"]

    _a = ((p.loc["COVID-19"].T - p.loc["COVID-19"].mean(1)) / p.loc["COVID-19"].std(1)).T
    _b = ((p.loc["Control"].T - p.loc["Control"].mean(1)) / p.loc["Control"].std(1)).T
    pdiff = _a - _b
    grid = clustermap(
        pdiff,
        center=0,
        cmap="RdBu_r",
        metric="correlation",
        cbar_kws=dict(label="Log-fold change"),
        row_cluster=False,
        # vmin=-5,
        # vmax=5,
    )
    grid.savefig(oprefix + "fold_change.clustermap.svg", **consts.figkws)

    ctoi = "Plasmacytoid dendritic cells"
    paths = pdiff.loc[ctoi].abs().sort_values().tail(8).index
    c = a.obs.loc[a.obs[cell_type_label] == ctoi]
    c["group"] = pd.Categorical(
        c["group"], categories=["Control", "COVID-19"], ordered=True
    )
    fig, _ = swarmboxenplot(data=c, x="group", y=paths)
    fig.savefig(oprefix + "fold_change.top_vars.swarmboxenplot.svg", **consts.figkws)

    # # Rank vs change plot
    fig, ax = plt.subplots(figsize=(6, 3))
    fc = pdiff.loc[ctoi]
    rank = fc.rank()
    ax.scatter(rank, fc)
    for v in rank.sort_values().tail(2).index:
        ax.text(rank.loc[v], fc.loc[v], s=v, ha="right")
    for v in rank.sort_values().head(2).index:
        ax.text(rank.loc[v], fc.loc[v], s=v, ha="left")
    ax.axhline(0, linestyle="--", color="grey")
    ax.set(xlabel="Pathway rank", ylabel="Log fold change (COVID-19 / Control)")
    fig.savefig(oprefix + "fold_change.rank_vs_value.svg", **consts.figkws)

    # # MA plot
    fig, ax = plt.subplots()
    mean = p.groupby(level=1).mean().loc[ctoi]
    fc = pdiff.loc[ctoi]
    ax.scatter(mean, fc)
    for v in mean.index:
        ax.text(mean.loc[v], fc.loc[v], s=v)
    fig.savefig(oprefix + "fold_change.rank_vs_value.svg", **consts.figkws)

    # TODO: look at specific gene expression from IFNalpha signature
    poi = "HALLMARK_INTERFERON_ALPHA_RESPONSE"
    genes = [g for g in sigs[poi] if g in a.raw.var_names]
    _a = a[a.obs[cell_type_label].isin([ctoi])]
    fig = sc.pl.heatmap(
        _a,
        var_names=genes,
        groupby=[cell_type_label, "group"],
        log=True,
        show_gene_labels=True,
        show=False,
    )["heatmap_ax"].figure
    fig.savefig(oprefix + "fold_change.signature.heatmap.svg", **consts.figkws)

    # _a = a[a.obs[cell_type_label].isin([ctoi])].raw.to_adata().to_df()[genes]
    # _a = _a.loc[:, _a.var() > 0]
    # grid = clustermap(_a, row_colors=a.obs[[cell_type_label, 'group']], config='z')

    # # by timing
    _a = a[a.obs["group"] == "COVID-19"]
    p = _a.obs.groupby(["timing_broad", cell_type_label])[signames].mean()
    grid = clustermap(p)
    grid.savefig(oprefix + "by_time_broad.clustermap.svg", **consts.figkws)

    kws = dict(
        center=0,
        cmap="RdBu_r",
        metric="correlation",
        vmin=-0.5,
        vmax=0.5,
        row_cluster=False,
    )
    pdiff = p.loc["Early"] - p.loc["Late"]
    grid = clustermap(pdiff, **kws)
    grid.savefig(oprefix + "by_time_broad.fold_changes.clustermap.svg", **consts.figkws)

    p = _a.obs.groupby(["timing_fine", cell_type_label])[signames].mean()
    grid = clustermap(p)
    grid.savefig(oprefix + "by_time_fine.clustermap.svg", **consts.figkws)

    kws = dict(center=0, cmap="RdBu_r", metric="correlation", vmin=-0.5, vmax=0.5)
    pdiff1 = (p.loc["Very early"] - p.loc["Early"]).rename_axis(
        index="Very early / Early"
    )
    grid = clustermap(pdiff1, **kws, cbar_kws=dict(label="Very early / Early"))
    grid.savefig(oprefix + "by_time_fine.fold_changes1.clustermap.svg", **consts.figkws)
    pdiff2 = (p.loc["Early"] - p.loc["Middle"]).rename_axis(index="Early / Middle")
    grid = clustermap(pdiff2, **kws, cbar_kws=dict(label="Early / Middle"))
    grid.savefig(oprefix + "by_time_fine.fold_changes2.clustermap.svg", **consts.figkws)
    pdiff3 = (p.loc["Middle"] - p.loc["Late"]).rename_axis(index="Middle / Late")
    grid = clustermap(pdiff3, **kws, cbar_kws=dict(label="Middle / Late"))
    grid.savefig(oprefix + "by_time_fine.fold_changes3.clustermap.svg", **consts.figkws)
    plt.close("all")

    kws = dict(center=0, cmap="RdBu_r", metric="correlation")
    for sig in signames:
        df = pd.DataFrame([pdiff1[sig], pdiff2[sig], pdiff3[sig]]).T
        df.columns = ["Very early / Early", "Early / Middle", "Middle / Late"]
        grid = clustermap(
            df, **kws, cbar_kws=dict(label="Log fold change"), col_cluster=False
        )
        grid.savefig(
            oprefix + f"by_time_fine.{sig}fold_changes.clustermap.svg", **consts.figkws
        )
    plt.close("all")

    # Macrophage IFN signature and inflammation
    immune_cell_types = a.obs.loc[
        a.obs["immune_status"] == "Immune", cell_type_label
    ].drop_duplicates()
    myeloid_cell_types = immune_cell_types[
        immune_cell_types.str.contains("acrophage|onocyte|lasmacytoid")
    ]

    p = a.obs.groupby(["group", cell_type_label])[signames].mean().dropna()
    pdiff = p.loc["COVID-19"] - p.loc["Control"]

    p = a.obs.groupby(["timing_broad", cell_type_label])[signames].mean().dropna()
    ptime = p.loc["Late"] - p.loc["Early"]

    grid = clustermap(
        pdiff.loc[myeloid_cell_types],
        cmap="RdBu_r",
        center=0,
        robust=True,
        metric="correlation",
        cbar_kws=dict(label="COVID-19 vs Control"),
    )
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=0)
    grid.savefig(
        oprefix + f"differential_disease_myeloid.clustermap.svg", **consts.figkws
    )
    plt.close(grid.fig)

    grid = clustermap(
        ptime.loc[myeloid_cell_types],
        cmap="RdBu_r",
        center=0,
        robust=True,
        metric="correlation",
        cbar_kws=dict(label="Late COVID-19 vs Early COVID-19"),
    )
    grid.ax_heatmap.set_yticklabels(grid.ax_heatmap.get_yticklabels(), rotation=0)
    grid.savefig(oprefix + f"differential_time_myeloid.clustermap.svg", **consts.figkws)
    plt.close(grid.fig)

    fig, axes = plt.subplots(len(myeloid_cell_types), 2, sharey=False)
    for ct, axs in zip(myeloid_cell_types, axes):
        for ax, d in zip(axs, [pdiff.loc[ct], ptime.loc[ct]]):
            rank = d.rank()
            ax.scatter(rank, d)
            for i in d.sort_values().head(3).index:
                ax.text(rank.loc[i], d.loc[i], s=i, ha="left")
            for i in d.sort_values().tail(3).index:
                ax.text(rank.loc[i], d.loc[i], s=i, ha="right")
            ax.axhline(0, linestyle="--", color="grey")
        axs[0].set(ylabel=ct)
    axes[0, 0].set(title="COVID-19 vs Control")
    axes[0, 1].set(title="Late COVID-19 vs Early COVID-19")
    fig.savefig(oprefix + f"differential_myeloid.rank_vs_value.svg", **consts.figkws)
    plt.close(fig)

    grid = clustermap(p.corr())

    pois = [
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "COVID-19 related inflammatory genes",
        "Fibrotic genes",
    ]
    _corrs = list()
    for ct in a.obs[cell_type_label].unique():
        c = a.obs.loc[a.obs[cell_type_label] == ct, pois].corr()
        _corrs.append(c.reset_index().melt(id_vars="index").assign(ct=ct))

    corrs = pd.concat(_corrs)
    c1 = (
        corrs.loc[(corrs["index"] == pois[0]) & (corrs["variable"] == pois[1])]
        .set_index("ct")["value"]
        .sort_values()
    )
    c2 = (
        corrs.loc[(corrs["index"] == pois[0]) & (corrs["variable"] == pois[2])]
        .set_index("ct")["value"]
        .sort_values()
    )
    c3 = (
        corrs.loc[(corrs["index"] == pois[1]) & (corrs["variable"] == pois[2])]
        .set_index("ct")["value"]
        .sort_values()
    )
    call = (
        c1.to_frame("IFNa -> Inflammation")
        .join(c2.to_frame("IFNa -> Fibrosis"))
        .join(c3.to_frame("Inflammation -> Fibrosis"))
    )
    call = call.sort_values(call.columns.tolist(), ascending=False)
    grid = clustermap(call, col_cluster=False, cmap="RdBu_r", center=0)
    grid = clustermap(
        call.loc[immune_cell_types], col_cluster=False, cmap="RdBu_r", center=0
    )

    fig, axes = plt.subplots(
        1, len(call.columns), figsize=(len(call.columns) * 7, 3.5), sharey=True
    )
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    for ax, c in zip(axes, call.columns):
        ax.bar(call[c].index, call[c], color=plt.get_cmap("RdBu_r")(norm(call[c])))
        ax.yaxis.tick_right()
        ax.set_xticklabels(call[c].index, rotation=90)
        ax.axhline(0, linestyle="--", color="grey")
        ax.set(ylabel=c)
    fig.savefig(oprefix + f"signature_correlation.barplot.svg", **consts.figkws)
    plt.close(fig)

    call2 = call.loc[immune_cell_types].sort_values(
        call.columns.tolist(), ascending=False
    )
    fig, axes = plt.subplots(
        1, len(call2.columns), figsize=(len(call2.columns) * 7, 3.5), sharey=True
    )
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    for ax, c in zip(axes, call2.columns):
        ax.bar(call2[c].index, call2[c], color=plt.get_cmap("RdBu_r")(norm(call2[c])))
        ax.yaxis.tick_right()
        ax.set_xticklabels(call2[c].index, rotation=90)
        ax.axhline(0, linestyle="--", color="grey")
        ax.set(ylabel=c)
    fig.savefig(
        oprefix + f"signature_correlation.barplot.immune_only.svg", **consts.figkws
    )
    plt.close(fig)

    cts = ["Alveolar macrophages", "Monocyte-derived macrophages", "Monocytes"]
    call = call.loc[cts].sort_values(call.columns.tolist(), ascending=False)
    fig, axes = plt.subplots(1, len(call.columns), figsize=(3, 2.5), sharey=True)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    for ax, c in zip(axes, call.columns):
        ax.bar(call[c].index, call[c], color=plt.get_cmap("RdBu_r")(norm(call[c])))
        ax.yaxis.tick_right()
        ax.set_xticklabels(call[c].index, rotation=90)
        ax.axhline(0, linestyle="--", color="grey")
        ax.set(ylabel=c)
    fig.savefig(oprefix + f"signature_correlation.fine.barplot.svg", **consts.figkws)
    plt.close(fig)


class consts:
    metadata_dir = Path("metadata").mkdir()
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
