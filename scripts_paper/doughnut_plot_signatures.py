import sys
sys.path.append("../")

import pandas as pd
from utils.plots import doughnut_plot_2_columns
from utils.read_in_utils import get_genes_per_hallmark, read_in_all_signatures
from utils.get_paths import FINAL_RESULTS_PATH

min_ngenes = 6
genes_per_hallmark = get_genes_per_hallmark(min_ngenes=min_ngenes)
n_genes_per_hm = pd.Series({k: len(v) for k, v in genes_per_hallmark.items()})
n_genes_per_hm.name = "n_genes"

signatures, index2biology = read_in_all_signatures(return_index2biology=True, min_ngenes=min_ngenes)

index2biology = {k: v for k, v in index2biology.items() if k in signatures.index}
index2biology = pd.Series(index2biology)
n_signs_per_hm = index2biology.value_counts()
n_signs_per_hm.name = "n_hallmarks"

plot_df = pd.merge(n_signs_per_hm, n_genes_per_hm, left_index=True, right_index=True)

doughnut_plot_2_columns(plot_df, hallmark_plot=True,
                        savename=FINAL_RESULTS_PATH + "doughnut_plot_hallmarks_at_least_%i.png"%min_ngenes)



