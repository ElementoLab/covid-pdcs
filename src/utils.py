"""
Utility functions used throughout the project.
"""

import matplotlib

from src.types import Figure


def rasterize_scanpy(fig: Figure) -> None:
    """
    Rasterize figure containing Scatter plots of single cells
    such as PCA and UMAP plots drawn by Scanpy.
    """
    import warnings

    with warnings.catch_warnings(record=False) as w:
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
