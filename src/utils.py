"""
Utility functions used throughout the project.
"""

import typing as tp

import pandas as pd
import matplotlib

from src.types import Figure, Path


def load_gene_signatures(msigdb: bool = False) -> tp.Dict[str, tp.List[str]]:
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


def rasterize_scanpy(fig: Figure) -> None:
    """
    Rasterize figure containing Scatter plots of single cells
    such as PCA and UMAP plots drawn by Scanpy.
    """
    import warnings

    with warnings.catch_warnings(record=False):
        warnings.simplefilter("ignore")
        yes_class = (
            matplotlib.collections.PathCollection,
            matplotlib.collections.LineCollection,
        )
        not_clss = (
            matplotlib.text.Text,
            matplotlib.axis.XAxis,
            matplotlib.axis.YAxis,
        )
        for axs in fig.axes:
            for __c in axs.get_children():
                if not isinstance(__c, not_clss):
                    if not __c.get_children():
                        if isinstance(__c, yes_class):
                            __c.set_rasterized(True)
                    for _cc in __c.get_children():
                        if not isinstance(_cc, not_clss):
                            if isinstance(_cc, yes_class):
                                _cc.set_rasterized(True)


def share_axis_scanpy(fig: Figure) -> None:
    for ax in fig.axes[::2]:
        ax.sharex(fig.axes[0])
        ax.sharey(fig.axes[0])
