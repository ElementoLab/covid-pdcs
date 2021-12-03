#!/usr/bin/env python

"""
Analysis of scRNA-seq data in light of pDC action in COVID-19.
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


# def load_delorey_adata():
#     b = sc.read("data/scrna/lung.h5ad")
#     # df = (
#     #     pd.read_csv("data/scrna/lung.scp.metadata.txt", sep="\t", skiprows=0)
#     #     .drop(0)
#     #     .convert_dtypes()
#     # )
#     return b
#     sc.pl.umap(b, color="SubCluster")
#     # Where are the promissed pDCs from Fig 2a and ED2i???


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


def msigdb_overlap():

    # MSigDB overlap
    gsl = open("h.all.v7.4.symbols.gmt").readlines()
    gsl = {x.split("\t")[0]: set(x.strip().split("\t")[2:]) for x in gsl}
    over = {
        (a_, b_): len(gsl[a_].intersection(gsl[b_])) / len(gsl[a_].union(gsl[b_]))
        for a_ in gsl
        for b_ in gsl
    }

    s = (
        pd.Series(over)
        .reset_index()
        .pivot_table(index="level_0", columns="level_1", values=0)
    )

    sns.heatmap(s, xticklabels=True, yticklabels=True)


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
    if not pdc_file.exists():
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

    res2 = res.query("pvals_adj < 0.01")
    res2["cluster"] = res2["cluster"].astype(int) + 1
    res2.to_excel(consts.results_dir / "DC_clustering.diff_expression.xlsx", index=False)

    # Show where IL3RA is
    p = res.query("cluster == '4'").sort_values("scores", ascending=False)
    p.head(20).to_csv("reviewer_table.csv")

    # coverage in pDCs

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


def compare_sig_diffs():
    n = load_adata()
    sigs = load_gene_signatures(msigdb=False)
    for sig in sigs:
        sc.tl.score_genes(n, sigs[sig], use_raw=True, score_name=sig)

    o = load_adata()
    sigs["COVID-19 related inflammatory genes"] += ["IFNB1", "IFNL1"]
    for sig in sigs:
        sc.tl.score_genes(o, sigs[sig], use_raw=True, score_name=sig)

    fig, axes = plt.subplots(1, 3, figsize=(4 * 3.3, 3), sharex=True, sharey=True)
    fig.suptitle(
        "Effect of removal of IFNB1 and IFNL1 from 'COVID-19 inflammation' signature."
    )
    sig = "COVID-19 related inflammatory genes"
    res = pg.corr(o.obs[sig].values, n.obs[sig].values)

    axes[0].scatter(
        o.obs[sig].values,
        n.obs[sig].values,
        c="grey",
        alpha=0.8,
        s=2,
        rasterized=True,
    )
    axes[0].set(title="All cells", ylabel="After removal")
    group_colors = sns.color_palette("tab10")
    for ax, group, c in zip(axes[1:], n.obs["group"].cat.categories[::-1], group_colors):
        sel = n.obs["group"] == group
        ax.scatter(
            o.obs.loc[sel, sig].values,
            n.obs.loc[sel, sig].values,
            c=c,
            alpha=0.8,
            s=2,
            rasterized=True,
        )
        ax.set(title=group)
        pg.corr(o.obs.loc[sel, sig].values, n.obs.loc[sel, sig].values)
    v = n.obs[sig].max()
    axes[2].set_xlabel("Before removal")
    for ax in axes:
        ax.plot((0, v), (0, v), linestyle="--", color="grey", alpha=0.5, zorder=-999)
    fig.savefig(
        consts.results_dir / "reviewer_figure.sig_genes_comparison.scatter.svg",
        **consts.figkws,
    )


def score_cell_types(a):
    import matplotlib

    for msigdb in [True, False]:
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

    _a = a[a.obs["group"] == "COVID-19"]
    fig = sc.pl.umap(_a, color=signames, vmin=vmin, vmax=vmax, show=False)[0].figure
    del _a
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / f"signature_enrichment{sign}.UMAP.COVID_cells.svg",
        **consts.figkws,
    )

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

    a.obs.groupby(cell_type_label)["nCount_RNA"].mean()
    counts = (
        a.obs.groupby(["patient", cell_type_label])
        .size()
        .reset_index()
        .pivot_table(index=cell_type_label, columns="patient", values=0)
    )
    counts.to_csv("cell_coverage.csv")

    fig = sc.pl.umap(a, color=cell_type_label, show=False).figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "UMAP.cell_types.svg", **consts.figkws)

    # Figure 3G
    cts = a.obs[cell_type_label].cat.categories
    a.obs["group"] = pd.Categorical(
        a.obs["group"], ordered=True, categories=["Control", "COVID-19"]
    )

    # Min max scale
    p = a.obs[signames]
    ref = p.groupby(a.obs[cell_type_label]).mean()
    p = ((p - ref.min()) / (ref.max() - ref.min())).join(
        a.obs[["group", cell_type_label]]
    )

    fig, axes = plt.subplots(3, len(cts), figsize=(1 * len(cts), 2 * 3), sharey="row")
    for ct, axs in zip(cts, axes.T):
        swarmboxenplot(
            data=p.loc[p[cell_type_label] == ct],
            x="group",
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
        for ax in axs:
            ax.set(xticklabels=[])
    fig.savefig(
        consts.results_dir / f"signature_enrichment.{cell_type_label}.barplot.svg",
        **consts.figkws,
    )

    _stats = list()
    for ct in cts:
        _stats += [
            swarmboxenplot(
                data=p.loc[p[cell_type_label] == ct],
                x="group",
                y=signames,
                swarm=False,
                boxen=False,
                bar=True,
                ax=axs,
                test=True,
            ).assign(cell_type=ct)
        ]
    stats = pd.concat(_stats)
    stats["p-cor"] = pg.multicomp(stats["p-unc"].values)[1]
    stats.to_csv(
        consts.results_dir / f"signature_enrichment.{cell_type_label}.test.csv",
        index=False,
    )

    fig, ax = plt.subplots(figsize=(8, 1))
    sns.heatmap(
        p.groupby([cell_type_label, "group"])[signames].mean().T,
        cmap="Reds",
        ax=ax,
        vmin=0,
        vmax=1,
    )
    fig.savefig(
        consts.results_dir / f"signature_enrichment.{cell_type_label}.heatmap.svg",
        **consts.figkws,
    )

    # Figure 3I
    # Correlate signatures per cell type
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].scatter(a.obs[signames[-1]], a.obs[signames[0]], s=2, alpha=0.2)
    axes[1].scatter(a.obs[signames[-1]], a.obs[signames[1]], s=2, alpha=0.2)

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

    # Figure 3H
    pois = [
        ("HALLMARK_INTERFERON_ALPHA_RESPONSE", "COVID-19 related inflammatory genes"),
        (
            "HALLMARK_INTERFERON_ALPHA_RESPONSE",
            "Fibrotic genes",
        ),
    ]
    _corrs = list()
    for ct in a.obs[cell_type_label].unique():
        for _q, _p in pois:
            _x = a.obs.loc[a.obs[cell_type_label] == ct]
            _a, _b = _x.loc[:, _q], _x.loc[:, _p]
            _r = pg.corr(_a, _b).assign(a=_q, b=_p)
            _r.index = [ct]
            _corrs.append(_r)
    corrs = pd.concat(_corrs)
    corrs["p-cor"] = pg.multicomp(corrs["p-val"].values)[1]
    c1 = corrs.loc[(corrs["a"] == pois[0][0]) & (corrs["b"] == pois[0][1])][
        "r"
    ].sort_values()
    c2 = corrs.loc[(corrs["a"] == pois[1][0]) & (corrs["b"] == pois[1][1])][
        "r"
    ].sort_values()
    call = c1.to_frame("IFNa -> Inflammation").join(c2.to_frame("IFNa -> Fibrosis"))
    call = call.loc[immune_cell_types].sort_values(call.columns.tolist(), ascending=False)

    fig, axes = plt.subplots(
        1, len(call.columns), figsize=(len(call.columns) * 5, 2.5), sharey=True
    )
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=1)
    for ax, c in zip(axes, call.columns):
        ax.bar(call[c].index, call[c], color=plt.get_cmap("RdBu_r")(norm(call[c])))
        ax.yaxis.tick_right()
        ax.set_xticklabels(call[c].index, rotation=90)
        ax.axhline(0, linestyle="--", color="grey")
        ax.set(ylabel=c)
    fig.savefig(
        oprefix + "signature_correlation.barplot.immune_only.svg", **consts.figkws
    )
    plt.close(fig)


def percent_infected_marcrophages_imc():
    positive_file = Path(
        "~/projects/archive/covid-imc/results/cell_type/gating.positive.pq"
    )
    q = pd.read_parquet(positive_file).drop("obj_id", axis=1)
    macs = q["cluster"].str.contains("Macrophage")
    mpos = q.loc[macs].groupby(["roi"]).sum()
    mtotal = q.loc[macs].groupby(["roi"]).size()
    mperc = (mpos.T / mtotal).T * 100

    # percent infected macrophages
    mperc.groupby(mperc.index.str.contains("COVID"))["SARSCoV2S1(Eu153)"].mean()
    # False    0.041761
    # True     7.824405
    # Name: SARSCoV2S1(Eu153), dtype: float64

    mpos = q.groupby(["roi", "cluster"]).sum()
    mtotal = q.groupby(["roi", "cluster"]).size()
    mperc = (mpos.T / mtotal).T * 100
    p = mperc.loc[mperc.index.get_level_values(0).str.contains("COVID")].reset_index()

    fig, ax = plt.subplots(figsize=(3, 3))
    s = p.query("cluster == '10 - Macrophages'")["SARSCoV2S1(Eu153)"]
    sns.histplot(s, ax=ax, bins=20, stat="percent")
    ax.axvline(s.mean(), color="brown", linestyle="--")
    ax.set(
        xlim=(-10, 110),
        xlabel="% SARS-CoV-2 Spike+\nmacrophages per image",
        ylabel=f"% images (n = {s.shape[0]})",
    )
    fig.savefig("macrophage_infection.per_image.pdf", bbox_inches="tight", dpi=300)


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
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
