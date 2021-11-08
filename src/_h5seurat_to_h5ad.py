#!/usr/bin/env python

"""
Convert RDS file to H5ad.
"""

"""
library("Seurat")
obj = readRDS("data/josefowicz/RNA_AllAssays_Final_Subset.rds")

obs = as.data.frame(obj@meta.data)
write.csv(obs, "data/josefowicz/metadata.csv")


write.csv(rownames(obj@assays$RNA@meta.features), "data/josefowicz/features.csv")
write.csv(rownames(obj@assays$SCT@scale.data), "data/josefowicz/sct_features.csv")

var_features = obj@assays$SCT@var.features
write.csv(var_features, "data/josefowicz/variable_features.csv")


pca = as.data.frame(obj@reductions$pca@cell.embeddings)
write.csv(pca, "data/josefowicz/reductions.pca.csv")
umap = as.data.frame(obj@reductions$umap@cell.embeddings)
write.csv(umap, "data/josefowicz/reductions.umap.csv")
harmony = as.data.frame(obj@reductions$harmony@cell.embeddings)
write.csv(harmony, "data/josefowicz/reductions.harmony.csv")
harmony.umap = as.data.frame(obj@reductions$harmony.umap@cell.embeddings)
write.csv(harmony.umap, "data/josefowicz/reductions.harmony.umap.csv")
ref.spca = as.data.frame(obj@reductions$ref.spca@cell.embeddings)
write.csv(ref.spca, "data/josefowicz/reductions.ref.spca.csv")
ref.umap = as.data.frame(obj@reductions$ref.umap@cell.embeddings)
write.csv(ref.umap, "data/josefowicz/reductions.ref.umap.csv")
"""

import pandas as pd
import AnnData
import anndata2ri
from rpy2.robjects import r

anndata2ri.activate()


obs = pd.read_csv("data/josefowicz/metadata.csv", index_col=0)
features = pd.read_csv("data/josefowicz/features.csv", index_col=0).squeeze()
sel_features = pd.read_csv("data/josefowicz/sct_features.csv", index_col=0).squeeze()
var_features = pd.read_csv("data/josefowicz/variable_features.csv", index_col=0).squeeze()

obj = r('readRDS("data/josefowicz/RNA_AllAssays_Final_Subset.rds")')
slots = list(obj.slotnames())
assays = obj.slots["assays"]
rna = assays[0]

ann_counts = AnnData(rna.slots["counts"].T)
ann_counts.var.index = features
# ann_counts.write("data/josefowicz/RNA_AllAssays_Final_Subset.counts.h5ad")

sct = assays[1]
ann_scaled = AnnData(sct.slots["scale.data"].T)
ann_scaled.obs = obs
ann_scaled.var.index = sel_features
ann_scaled.var["highly_variable"] = ann_scaled.var.index.isin(var_features)
ann_scaled.raw = ann_counts


reds = ["pca", "umap", "harmony", "harmony.umap", "ref.spca", "ref.umap"]
for red in reds:
    ann_scaled.obsm[red] = pd.read_csv(
        f"data/josefowicz/reductions.{red}.csv", index_col=0
    )
ann_scaled.write("data/josefowicz/RNA_AllAssays_Final_Subset.scaled.h5ad")
