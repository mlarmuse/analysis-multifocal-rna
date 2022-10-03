"""
Script to plot PCA and add the centroids on the plot
"""

import sys
sys.path.append("../")

from utils.plots import plot_scatter, sort_ids_by_patient_sample_type_number
import pandas as pd
from utils.read_in_utils import read_in_all_signatures, get_expression
from utils.CentroidClassifier import CentroidClassifier, SpearmanPCA
import numpy as np

exp = get_expression(exp_df=None,
                     calc_zscore=False,
                     reference_tissue="NL",
                     collection_genes_to_keep="all",
                     log_tf=False,
                     remove_zv_genes=False)

exp = exp.transpose()

signatures, index2biology = read_in_all_signatures(return_index2biology=True)
all_features = signatures.index.values

probs_TCGA = {}
probs_TCGA_patient = {}

signature = "Focal adhesion" #"HALLMARK_FATTY_ACID_METABOLISM"  # "Mitochondrial: Complex IV"#

genes = np.intersect1d(exp.columns.values, signatures.loc[signature].astype(str))

spPCA = SpearmanPCA(n_components=2)
plot_df = spPCA.fit_transform(exp[genes].values)
plot_df = pd.DataFrame(plot_df, index=exp.index, columns=["PC 1", "PC 2"])

y = np.array([2 if ("LN" in s.split("_")[-1]) else 1 if ("RP" in s.split("_")[-1]) else 0
              for s in exp.index]).astype(int)

cc = CentroidClassifier()
cc.fit(exp[genes].values, y_train=y)
centroid_dict = dict(zip(["AN", "PT", "MLN"], spPCA.transform(cc.centroids)))

sort_idx = sort_ids_by_patient_sample_type_number(plot_df.index.values)
plot_df = plot_df.loc[sort_idx]
plot_scatter(plot_df,
             centroids=centroid_dict,
             plot_arrows=True,
             savename="PCA_%s.svg" % signature)
