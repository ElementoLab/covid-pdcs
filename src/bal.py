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
# # - BAL:
# # - [ ] plots on signature correlations
# # - [ ] do diffusion on pDCs
# # - [ ] enrichment seperately for Mild/Severe
# # - [ ] enrichment seperately for Mild/Severe
# # - [ ] histogram of disease states across DC1
# # - [ ] histogram of signature enrichemnt across DC1


# # infected macrophages?


def main():
    a = get_anndata("phenotyped")
    processing(a)
    phenotyping(a)
    score_cell_types(a)
    macrophage_focus(a)


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

    # sc.external.pp.harmony_integrate(a, key="patient")
    # sc.pp.neighbors(a, use_rep='X_pca_harmony')
    # sc.tl.umap(a)

    # fig = sc.pl.umap(a, color=['cell_type_label', 'patient', 'disease_severity'], show=False)[0].figure
    # rasterize_scanpy(fig)
    # fig.savefig(consts.results_dir / "clustering.cell_type_label.on_harmony.svg", **consts.figkws)
    # plt.close(fig)


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
    genes = [g for g in consts.pbmc_markers if g in a.raw.var.index.tolist()]
    vmaxes = [
        percentile(a.raw.X[:, a.var.index == g].todense().squeeze(), 95) for g in genes
    ]
    a.uns["disease_severity_colors"] = ["#1f77b4", "#ff7f0e", "#d62728"]
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


def compare_sig_diffs():
    sig = consts.soi[1]

    n = get_anndata("phenotyped")
    sigs = load_gene_signatures(msigdb=False)
    for sig in sigs:
        sc.tl.score_genes(n, sigs[sig], use_raw=True, score_name=sig)

    o = get_anndata("phenotyped")
    # add 'wrong genes'
    sigs["COVID-19 related inflammatory genes"] += ["IFNB1", "IFNL1"]
    for sig in sigs:
        sc.tl.score_genes(o, sigs[sig], use_raw=True, score_name=sig)

    disease_severity_colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    o.uns["disease_severity_colors"] = disease_severity_colors

    fig, axes = plt.subplots(1, 4, figsize=(4 * 3.3, 3), sharex=True, sharey=True)
    fig.suptitle(
        "Effect of removal of IFNB1 and IFNL1 from 'COVID-19 inflammation' signature."
    )
    axes[0].scatter(
        o.obs[sig].values,
        n.obs[sig].values,
        c="grey",
        alpha=0.8,
        s=2,
        rasterized=True,
    )
    axes[0].set(title="All cells", ylabel="After removal")
    for ax, group, c in zip(
        axes[1:], n.obs["disease_severity"].cat.categories, disease_severity_colors
    ):
        sel = n.obs["disease_severity"] == group
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

    means = pd.Series(o.raw.to_adata().todense().mean(0), index=o.var.index)
    genes = ["IFNB1", "IFNL1"]
    x = o.raw[:, genes].X.todense()

    sc.pl.heatmap(
        o, var_names=genes, groupby="cell_type_label", log=True, standard_scale="obs"
    )

    xm = (
        o.raw[:, genes]
        .to_adata()
        .to_df()
        .join(o.obs["cell_type_label"])
        .groupby("cell_type_label")
        .mean()
    )
    cm = o.obs.groupby("cell_type_label")["n_counts"].mean()
    (xm.T / cm).T * 1e6


def score_cell_types(a):
    cell_type_label = "cell_type_label"

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
            sigs = list(sigs.keys())
        else:
            sigs = load_gene_signatures(msigdb=msigdb)
            sigs = list(sigs.keys())
            a.obs = a.obs.join(pd.read_csv(sig_f, index_col=0))
        signames += sigs

    # Plot signatures at single-cell (UMAP overlay)
    vmin = 0
    vmax = [percentile(a.obs[sig], 95) for sig in consts.soi]
    for cmap in ["viridis", "magma", "inferno", "Reds"]:
        fig = sc.pl.umap(
            a, color=consts.soi, vmin=vmin, vmax=vmax, show=False, cmap=cmap
        )[0].figure
        rasterize_scanpy(fig)
        fig.savefig(
            consts.results_dir / f"signature_enrichment{sign}.UMAP.all_cells.{cmap}.svg",
            **consts.figkws,
        )

    _a = a[a.obs["disease"] == "COVID-19"]
    fig = sc.pl.umap(_a, color=consts.soi, vmin=vmin, vmax=vmax, show=False)[0].figure
    del _a
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / f"signature_enrichment{sign}.UMAP.COVID_cells.svg",
        **consts.figkws,
    )

    fig = sc.pl.umap(a, color=cell_type_label, show=False).figure
    rasterize_scanpy(fig)
    fig.savefig(consts.results_dir / "UMAP.cell_types.svg", **consts.figkws)

    # Plot signatures agregated by cell type
    sig_means = (
        a.obs.groupby(["patient", cell_type_label, "disease_severity"])[signames]
        .mean()
        .groupby(level=[cell_type_label, "disease_severity"])
        .mean()
    )
    # # replace non existing with jitter
    _m = sig_means.isnull().all(1)
    sig_means.loc[_m] = np.random.random(len(signames)) * 1e-10

    sig_diff = pd.concat(
        [
            (
                sig_means.loc[:, v, :].reset_index(level=1, drop=True)
                - sig_means.loc[:, "Control", :].reset_index(level=1, drop=True)
            )
            .assign(disease_severity=v)
            .set_index("disease_severity", append=True)
            for v in a.obs["disease_severity"].cat.categories[1:]
        ]
    ).sort_index()

    grid = clustermap(
        sig_diff.T,
        col_cluster=False,
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 9),
    )
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.diff.svg",
        **consts.figkws,
    )

    grid = clustermap(
        sig_means.fillna(0).T,
        col_cluster=False,
        col_colors=sig_means.index.to_frame()[["disease_severity"]],
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 9),
    )
    grid.ax_heatmap.set_xticks(range(0, sig_means.shape[0], 3))
    grid.ax_heatmap.set_xticklabels(sig_means.index.levels[0])
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.svg",
        **consts.figkws,
    )

    grid = clustermap(
        sig_means.fillna(0).T.loc[consts.soi],
        col_cluster=False,
        col_colors=sig_means.index.to_frame()[["disease_severity"]],
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 3),
    )
    grid.ax_heatmap.set_xticks(range(0, sig_means.shape[0], 3))
    grid.ax_heatmap.set_xticklabels(sig_means.index.levels[0])
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.specific_sigs.svg",
        **consts.figkws,
    )

    grid = clustermap(
        sig_means.fillna(0).T,
        col_cluster=False,
        col_colors=sig_means.index.to_frame()[["disease_severity"]],
        xticklabels=True,
        cmap="RdBu_r",
        robust=True,
        center=0,
        figsize=(12, 9),
        z_score=0,
    )
    grid.ax_heatmap.set_xticks(range(0, sig_means.shape[0], 3))
    grid.ax_heatmap.set_xticklabels(sig_means.index.levels[0])
    grid.fig.savefig(
        consts.results_dir
        / "signature_enrichment.all_sigs.all_cell_types.clustermap.z_score.svg",
        **consts.figkws,
    )

    # means = a.obs.groupby(["disease", cell_type_label])[consts.soi].mean().loc["COVID-19"]

    cts = a.obs[cell_type_label].cat.categories
    fig, axes = plt.subplots(
        len(consts.soi),
        len(cts),
        figsize=(len(cts), len(consts.soi)),
        sharey="row",
        sharex=True,
    )
    for ct, axs in zip(cts, axes.T):
        _ = swarmboxenplot(
            data=a.obs.loc[a.obs[cell_type_label] == ct],
            x="disease_severity",
            y=consts.soi,
            swarm=False,
            boxen=False,
            bar=True,
            ax=axs,
            test=False,
        )
        axs[0].set(title=ct)
        for ax in axs[1:]:
            ax.set(title="")
    for ax, sig in zip(axes[:, 0], consts.soi):
        ax.set(ylabel=sig)
    fig.savefig(
        consts.results_dir / f"signature_enrichment.{cell_type_label}.barplot.svg",
        **consts.figkws,
    )

    # Signature correlation
    corrs = (
        a[a.obs["disease_severity"] == "Mild"]
        .obs.groupby(cell_type_label)[consts.soi]
        .corr()
    )

    corr = corrs.reset_index().pivot_table(index=cell_type_label, columns="level_1")
    corr = corr.loc[:, ~(corr == 1).all()]
    corr = corr.loc[:, corr.sum().drop_duplicates().index]
    grid = clustermap(corr.T, cmap="RdBu_r", center=0, figsize=(10, 3), row_cluster=False)
    grid.fig.savefig(
        consts.results_dir
        / f"signature_enrichment.correlation.{cell_type_label}.heatmap.immune.svg",
        **consts.figkws,
    )

    cts = a.obs[cell_type_label].unique()
    states = a.obs["disease_severity"].cat.categories
    colors = np.asarray(sns.color_palette())[[0, 1, 3]]
    _a = "HALLMARK_INTERFERON_ALPHA_RESPONSE"
    _b = "COVID-19 related inflammatory genes"

    a.obs[_a + "_plt"] = a.obs[_a] + abs(a.obs[_a].min())
    a.obs[_b + "_plt"] = a.obs[_b] + abs(a.obs[_b].min())
    for ct in cts:
        fig, axes = plt.subplots(
            1, len(states), sharex=True, sharey=True, figsize=(3 * len(states), 1.5)
        )

        _tpcs = dict()
        for state in states:
            _tpcs[state] = a.obs.loc[
                (a.obs[cell_type_label] == ct) & (a.obs["disease_severity"] == state)
            ]
        n = min(tuple(map(len, _tpcs.values())))
        _tpcs = {k: v.sample(n=n) for k, v in _tpcs.items()}

        comp = pd.concat(_tpcs.values())
        at = comp[_a + "_plt"].mean()
        bt = comp[_b + "_plt"].mean()

        for i, (state, ax) in enumerate(zip(states, axes)):
            _tpc = _tpcs[state]
            if _tpc.empty:
                continue
            _tpc[[_a + "_plt", _b + "_plt"]] += 0.5
            r = pg.corr(_tpc[_a + "_plt"], _tpc[_b + "_plt"]).squeeze()
            ap = ((_tpc[_a + "_plt"] > at).sum() / _tpc[_a + "_plt"].size) * 100
            bp = ((_tpc[_b + "_plt"] > bt).sum() / _tpc[_b + "_plt"].size) * 100
            cp = (
                ((_tpc[_a + "_plt"] > at) & (_tpc[_b + "_plt"] > bt)).sum()
                / _tpc[_b + "_plt"].size
            ) * 100

            ax.set(
                title=f"r = {r['r']:.3f}; p = {r['p-val']:.2e};\ndouble-positive = {cp:.1f}%"
            )

            sns.regplot(
                x=_tpc[_a + "_plt"],
                y=_tpc[_b + "_plt"],
                ax=ax,
                scatter=False,
                color=colors[i],
            )
            ax.scatter(
                _tpc[_a + "_plt"],
                _tpc[_b + "_plt"],
                alpha=0.2,
                s=2,
                color=colors[i],
                rasterized=True,
            )
            ax.axvline(at, linestyle="--", linewidth=0.2, color=colors[i])
            ax.axhline(bt, linestyle="--", linewidth=0.2, color=colors[i])
            ax.set(xlabel=_a, ylabel=_b)
        ct = ct.replace("/", "-")

        for ax in axes:
            vmin = comp[_b + "_plt"].min()
            vmin -= vmin * 0.25
            vmin = None if pd.isnull(vmin) else vmin
            vmax = comp[_b + "_plt"].max()
            vmax += vmax * 0.1
            vmax = None if pd.isnull(vmax) else vmax
            ax.set_ylim(bottom=vmin, top=vmax)

        fig.savefig(
            consts.results_dir
            / f"signature_enrichment.correlation.{cell_type_label}.selected_{ct}.scatter.svg",
            **consts.figkws,
        )
        plt.close(fig)

        for ax in axes:
            ax.loglog()
            ax.set_xlim(left=10)
            vmin = comp[_b + "_plt"].min()
            vmin -= vmin * 0.1
            vmax = comp[_b + "_plt"].max()
            vmax += vmax * 0.1
            vmax = None if pd.isnull(vmax) else vmax
            ax.set_ylim(bottom=max(1, vmin), top=vmax)
        fig.savefig(
            consts.results_dir
            / f"signature_enrichment.correlation.{cell_type_label}.selected_{ct}.scatter.log.svg",
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


def macrophage_focus(a: AnnData):
    a.uns["disease_severity_colors"] = ["#1f77b4", "#ff7f0e", "#d62728"]
    macs = a[a.obs["cell_type_label"] == "Macrophages"]

    sc.tl.diffmap(macs)
    fig = sc.pl.diffmap(macs, color="disease_severity", show=False, s=5).figure
    rasterize_scanpy(fig)
    fig.savefig(
        consts.results_dir / "macrophage_diversity.diffmap.svg",
        **consts.figkws,
    )

    # macs.var['xroot'] = macs[dc1.idxmin(), :].X.squeeze()
    # sc.tl.dpt(macs)
    # dc1 = macs.obs['dpt_pseudotime'].rename("DC1")

    dc1 = pd.Series(macs.obsm["X_diffmap"][:, 1], index=macs.obs.index, name="DC1")
    fig, axes = plt.subplots(6, 1, figsize=(4, 4), sharex=True)
    for i, (ax, group) in enumerate(
        zip(axes.flat, macs.obs["disease_severity"].cat.categories)
    ):
        sel = macs.obs["disease_severity"] == group
        sns.distplot(
            dc1[sel],
            hist=False,
            norm_hist=False,
            ax=ax,
            color=macs.uns["disease_severity_colors"][i],
        )
        ax.set_ylim(top=400)
    p = macs.obs[consts.soi].join(dc1)
    # p = p.loc[macs.obs['disease_severity'] != 'Mild']
    p["DC1_rank"] = p["DC1"].rank()
    p["bins"] = pd.cut(p["DC1"], 50)
    pp = p.groupby("bins").mean()
    pp["size"] = p.groupby("bins").size()
    # pp = pp.loc[pp["size"] > 100]
    pp.index = [np.mean([i.left, i.right]) for i in pp.index]
    # pp = p.rolling(window=500, min_periods=100).median().resample().sort_index()
    for i, (ax, sig) in enumerate(zip(axes[3:], consts.soi)):
        ax.scatter(pp.index, pp[sig], c=pp[sig], cmap="inferno")
    fig.savefig(
        consts.results_dir / "macrophage_diversity.diffmap.distributions_along_path.svg",
        **consts.figkws,
    )

    sc.tl.rank_genes_groups(
        macs,
        groupby="disease_severity",
        method="t-test_overestim_var",
        reference="Control",
    )
    _diffs = [
        sc.get.rank_genes_groups_df(macs, "Mild").assign(group="Mild/Healthy"),
        sc.get.rank_genes_groups_df(macs, "Severe").assign(group="Severe/Healthy"),
    ]
    sc.tl.rank_genes_groups(
        macs,
        groupby="disease_severity",
        method="t-test_overestim_var",
        reference="Mild",
    )
    _diffs += [
        sc.get.rank_genes_groups_df(macs, "Severe").assign(group="Severe/Mild"),
    ]
    diffs = pd.concat(_diffs)

    diff_genes = diffs.loc[diffs["pvals_adj"] == 0].set_index(["group", "names"])

    import matplotlib.pyplot as plt
    from matplotlib_venn import venn3

    fig, ax = plt.subplots()
    venn3(
        [set(diff_genes.loc[c].index) for c in diff_genes.index.levels[0]],
        diff_genes.index.levels[0],
        ax=ax,
    )
    fig.savefig(
        consts.results_dir
        / f"macrophage_diversity.differential_expression.venn_diagram.svg",
        **consts.figkws,
    )
    # diff_genes = [g for g in diff_genes if g in macs.var.index]

    fig = sc.pl.heatmap(
        macs,
        var_names=diff_genes.index.get_level_values(1),
        groupby="disease_severity",
        log=True,
        standard_scale="var",
        show_gene_labels=True,
        show=False,
        figsize=(10, 2),
    )["heatmap_ax"].figure
    fig.savefig(
        consts.results_dir
        / "macrophage_diversity.differential_expression.top_markers.heatmap.svg",
        **consts.figkws,
    )

    import gseapy

    diff_genes = diffs.set_index("names").groupby("group")["scores"].nlargest(100)
    res = pd.concat(
        [
            gseapy.enrichr(
                diff_genes.loc[group].index.tolist(), consts.gene_set_libraries
            ).results.assign(group=group)
            for group in diff_genes.index.levels[0]
        ]
    )
    res["mlogp"] = -np.log10(res["P-value"])

    for measure in ["Combined Score", "mlogp"]:
        groups = diff_genes.index.levels[0]
        fig, axes = plt.subplots(
            len(consts.gene_set_libraries), len(groups), squeeze=False, figsize=(8, 2)
        )
        for axs, gsl in zip(axes, consts.gene_set_libraries):
            for ax, group in zip(axs, groups):
                p = (
                    res.query(f"Gene_set == '{gsl}' & group == '{group}'")
                    .sort_values(measure, ascending=False)
                    .head(10)
                )
                sns.barplot(
                    data=p,
                    x=measure,
                    y="Term",
                    orient="horiz",
                    ax=ax,
                    color=sns.color_palette("inferno")[0],
                )
                ax.set_title(group)
            axs[0].set_ylabel(gsl)
        fig.savefig(
            consts.results_dir
            / f"macrophage_diversity.differential_expression.{measure}.enrichment.barplot.svg",
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
        # "TPPP3",
        # "KRT18",
        "CD3D",
        "CD8A",
        "CD4",
        "NKG7",
        "CD68",
        # "FCGR3B",
        "CD1C",
        "CLEC4C",
        # "CLEC9A",
        # "TPSB2",
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
    soi = [
        "HALLMARK_INTERFERON_ALPHA_RESPONSE",
        "COVID-19 related inflammatory genes",
        "Fibrotic genes",
    ]
    figkws = dict(bbox_inches="tight", dpi=300)

    gene_set_libraries = [
        "GO_Biological_Process_2015",
        # "ChEA_2015",
        # "KEGG_2016",
        # "ESCAPE",
        # "Epigenomics_Roadmap_HM_ChIP-seq",
        # "ENCODE_TF_ChIP-seq_2015",
        # "ENCODE_and_ChEA_Consensus_TFs_from_ChIP-X",
        # "ENCODE_Histone_Modifications_2015",
        # "OMIM_Expanded",
        # "TF-LOF_Expression_from_GEO",
        # "Gene_Perturbations_from_GEO_down",
        # "Gene_Perturbations_from_GEO_up",
        # "Disease_Perturbations_from_GEO_down",
        # "Disease_Perturbations_from_GEO_up",
        # "Drug_Perturbations_from_GEO_down",
        # "Drug_Perturbations_from_GEO_up",
        # "WikiPathways_2016",
        # "Reactome_2016",
        # "BioCarta_2016",
        # "NCI-Nature_2016",
        # "BioPlanet_2019",
    ]


from functools import partial

clustermap = partial(clustermap, dendrogram_ratio=0.1)


def percentile(x, i):
    import numpy

    try:
        return numpy.percentile(x, i)
    except IndexError:
        return 0


if __name__ == "__main__" and "get_ipython" not in locals():
    import sys

    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(1)
