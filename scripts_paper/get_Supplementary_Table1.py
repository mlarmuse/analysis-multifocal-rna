"""
Creates supplementary file 1, with all important extra information regarding the study.
"""

import sys
sys.path.append("../")

from utils.get_paths import FINAL_RESULTS_PATH, INPUT_PATH
from utils.read_in_utils import get_clinical_info_UZ, get_sample_purities
import pandas as pd
import numpy as np


def write_to_excel(df, sheet_name, file=FINAL_RESULTS_PATH + "Table_S1.xlsx",
                   mode="a",
                   if_sheet_exists="replace"):
    with pd.ExcelWriter(file,
                        mode=mode,
                        engine="openpyxl",
                        if_sheet_exists=if_sheet_exists)\
            as writer:
        df.to_excel(writer, sheet_name=sheet_name)


# first we get the clinical_df and the purity data:
clin_df = get_clinical_info_UZ()
purity_df = get_sample_purities()
purity_df.name = "EPIC estimated purity"

total_clin_df = pd.merge(clin_df, purity_df, left_index=True, right_index=True, how="inner")

write_to_excel(total_clin_df, sheet_name="Clinical_Data", mode="w")

# Then we write the results from the UZ cross-validation
uz_classification = pd.read_csv(FINAL_RESULTS_PATH + "Classification_table_UZ_Ghent.csv", index_col=0)
write_to_excel(uz_classification, sheet_name="Tissue_classification_UZ")

# Then we write the MLN centroid scores on TCGA
TCGA_MLN_scores = pd.read_csv(FINAL_RESULTS_PATH + "Results_MLN_centroid_TCGA_fixed_seed.csv", index_col=0)
write_to_excel(TCGA_MLN_scores[["AUC", "pval", "FDR", "Hallmark"]], sheet_name="MLN_centroid_TCGA")

# Next, we add the complete results of the seeding test:
seeding_scores = pd.read_csv(FINAL_RESULTS_PATH + "table_top_signaures_seeding_test_confirmed_patients.csv",
                             index_col=0)
write_to_excel(seeding_scores, sheet_name="Seeding_test")

# Add information about the number of COSMIC variants per lesion:
mut_counts_per_patient = pd.read_csv(INPUT_PATH + "mut_per_patient_counts__depth6.csv", index_col=0)
mut_counts_per_patient.name = "Number_of_retained_variants"
write_to_excel(mut_counts_per_patient, sheet_name="Mutation_count_per_patient")

# Add p-value tables per patient:
# this is written to a CSV and can be merged manually later on:

variant_scores_per_patient = pd.read_csv(INPUT_PATH + "RP_2_MLN_pvals_clear_depth%i.csv" % 6,
                                         index_col=0)
outfile = FINAL_RESULTS_PATH + "pat_scores_RP2MLN.csv"

r_patient_ids = np.array([s.split("_")[1] for s in variant_scores_per_patient.index.values])
c_patient_ids = np.array([s.split("_")[1] for s in variant_scores_per_patient.columns.values])
patients = np.array([s.split("_")[1] for s in variant_scores_per_patient.index.values])
uniq_patients = np.unique(r_patient_ids)

for i, pat in enumerate(uniq_patients):
    pat_var_mat = variant_scores_per_patient.loc[r_patient_ids == pat].loc[:, c_patient_ids == pat]

    if i == 0:
        f = open(outfile, "w")
        f.write(pat)
        f.close()

    else:
        f = open(outfile, "a")
        f.write("\n")
        f.write(pat)
        f.close()

    pat_var_mat.to_csv(outfile, mode="a")
