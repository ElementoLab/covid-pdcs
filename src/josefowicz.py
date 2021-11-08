#!/usr/bin/env python

"""
Analysis of scRNA-seq data from BALF in light of pDC action in COVID-19.
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


class consts:
    data_dir = Path("data").mkdir()
    metadata_dir = Path("metadata").mkdir()
    results_dir = (Path("results") / "josefowicz").mkdir()
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
