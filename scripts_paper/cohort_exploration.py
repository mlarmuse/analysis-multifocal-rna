"""
Creates the PCA plot that describes the cohort
"""

import sys
sys.path.append("../")

import pandas as pd
import numpy as np
from utils.plots import sns_scatter_with_contour
from utils.read_in_utils import get_expression, get_signature_genes
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

topn = 5000
exp = get_expression()
exp = exp.transpose()

signature_genes = get_signature_genes(exp=exp, min_ngenes=6, collection_name="all")
common_genes = np.intersect1d(exp.columns.values, signature_genes)
exp = exp[common_genes]

# top_var_genes = exp.var(axis=0).sort_values(ascending=False).index.values
# exp = exp[top_var_genes]

ss = StandardScaler()
X = ss.fit_transform(exp.values)

pc = PCA(n_components=2)
X_red = pc.fit_transform(X=X)
plot_df = pd.DataFrame(X_red, index=exp.index, columns=["PC 1", "PC 2"])
pc.explained_variance_ratio_
sns_scatter_with_contour(plot_df, marginal_kws=dict(common_norm=False), save_name="PCA_plot_new")