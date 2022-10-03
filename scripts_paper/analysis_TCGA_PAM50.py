import sys
sys.path.append("../")

import pandas as pd
from utils.read_in_utils import INPUT_PATH
import numpy as np
from scipy.stats import chi2_contingency, fisher_exact

nodal_status = pd.read_csv(INPUT_PATH + "nodal_status_PRAD_broad_processed.txt", sep="\t", index_col=0)
pam50_labels = INPUT_PATH + "pam50_TCGA.csv"

pam50_TCGA = pd.read_csv(pam50_labels, index_col=0)
pam50_TCGA.index = [s.replace(".", "-") for s in pam50_TCGA.index]

pam50_TCGA = pam50_TCGA[["Basal", "LumA", "LumB"]]
pam50_TCGA = pam50_TCGA/np.sum(pam50_TCGA.values, axis=1, keepdims=True)

pam50_labels = pam50_TCGA.columns.values[np.argmax(pam50_TCGA.values, axis=1)]
pam50_labels = pd.Series(pam50_labels, index=pam50_TCGA.index, name="PAM50")

data_df = pd.merge(nodal_status, pam50_labels, left_index=True, right_index=True)
cont_table = pd.crosstab(data_df["PAM50"], data_df["ajcc_pathologic_n"])

pd.DataFrame(cont_table)

cont_table = cont_table.values
chi2_contingency(cont_table)

cont_table2 = np.array([cont_table[0, :] + cont_table[1, :],  cont_table[2, :]])

fisher_exact(cont_table2, alternative="greater")
