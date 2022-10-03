import sys
sys.path.append("../")

import pandas as pd
from utils.get_paths import INPUT_PATH
from utils.read_in_utils import get_TCGA_data_as_one, get_TCGA_data
import numpy as np
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from utils.get_paths import FINAL_RESULTS_PATH

# TODO: check how other papers dealt with PSA taken years after resection
# TODO: create input table for partin prediction


def fisherExact(v, y, th=1e-15, alternative='two-sided'):
    table = np.array([[np.sum((v < th) & (y == 0)), np.sum((v < th) & (y != 0))],
                      [np.sum((v >= th) & (y == 0)), np.sum((v >= th) & (y != 0))]])
    return fisher_exact(table, alternative=alternative)[1]


def remove_nans(arr):
    arr = np.asarray(arr)
    otype = arr.dtype

    mask = np.asarray([s != "nan" for s in arr.astype(str)])

    return arr[mask].astype(otype), mask


def process_row_float(data, groups, decimal_places=2):

    data_name = data.name
    data = data.values
    data, mask = remove_nans(data)
    groups = groups[mask]
    data = data.astype(float)

    odf = {"all": "%0.1f (%0.1f)" % (data.mean(), np.median(data))}

    uniq_groups = np.unique(groups)

    for group in uniq_groups:
        data_group = data[groups == group]
        odf[group] = "%0.1f (%0.1f)" % (data_group.mean(), np.median(data_group))

    if len(uniq_groups) == 2:
        pval = mannwhitneyu(data[groups == uniq_groups[0]], data[groups == uniq_groups[1]])[1]

        odf["P-value"] = pval

    median_series = pd.Series(odf)
    median_series.name = "%s: Mean (median)" % data_name

    odf = {"all": "%0.1f - %0.1f" % (data.min(), data.max())}

    uniq_groups = np.unique(groups)

    for group in uniq_groups:
        data_group = data[groups == group]
        odf[group] = "%0.1f - %0.1f" % (data_group.min(), data_group.max())

    odf["P-value"] = np.nan

    range_series = pd.Series(odf)
    range_series.name = "%s: Min-Max" % data_name

    return pd.concat([median_series, range_series], axis=1).transpose()


def process_row_categorical(data, groups):
    data = data.name + ": " + data.astype(str)
    data = data.values
    data, mask = remove_nans(data)
    groups = groups[mask]

    uniq_groups = np.unique(groups)

    cont_table = pd.crosstab(index=data, columns=groups)
    cont_table["all"] = cont_table.sum(axis=1)

    pval = chi2_contingency(cont_table.values)[1]
    cont_table["P-value"] = np.nan
    cont_table.iloc[0, -1] = pval

    return cont_table[["all"] + list(uniq_groups) + ["P-value"]]


def process_row(data, groups, mode="categorical"):

    if mode == "categorical":
        df = process_row_categorical(data, groups)

    else:
        df = process_row_float(data, groups)

    return df


if __name__ == "__main__":
    source = "broad"

    file = INPUT_PATH + "PRAD.clin.merged.txt"

    clin_table = pd.read_csv(file, sep="\t", index_col=0, header=None)
    clin_table.columns = [pat.upper() for pat in clin_table.loc["patient.bcr_patient_barcode"]]

    exp_tumor, exp_normal, nodal_status = get_TCGA_data(nodal_source="broad")
    common_samples = np.intersect1d(exp_tumor.columns.values, clin_table.columns.values)
    clin_table = clin_table[common_samples]

    row_of_interest = ["patient.age_at_initial_pathologic_diagnosis",
                       "patient.other_dx",
                       "patient.stage_event.psa.psa_value",
                       "patient.stage_event.psa.days_to_psa",
                       "patient.number_of_lymphnodes_positive_by_he",
                       "patient.stage_event.gleason_grading.gleason_score",
                       "patient.stage_event.gleason_grading.primary_pattern",
                       "patient.stage_event.gleason_grading.secondary_pattern",
                       "patient.stage_event.tnm_categories.pathologic_categories.pathologic_n",
                       "patient.stage_event.tnm_categories.pathologic_categories.pathologic_t",
                       "patient.stage_event.tnm_categories.clinical_categories.clinical_t"
                       ]

    clin_description = clin_table.loc[row_of_interest].transpose()

    rename_dict = {"patient.age_at_initial_pathologic_diagnosis": "age",
                   "patient.other_dx": "prior malignancy",
                   "patient.stage_event.psa.psa_value": "psa",
                   "patient.stage_event.psa.days_to_psa": "days to psa",
                   "patient.number_of_lymphnodes_positive_by_he": "n_lymph_nodes",
                   "patient.stage_event.gleason_grading.gleason_score": "Gleason Score (GS)",
                   "patient.stage_event.gleason_grading.primary_pattern": "GS primary",
                   "patient.stage_event.gleason_grading.secondary_pattern": "GS secondary",
                   "patient.stage_event.tnm_categories.pathologic_categories.pathologic_n": "pathologic n",
                   "patient.stage_event.tnm_categories.pathologic_categories.pathologic_t": "pathologic t",
                   "patient.stage_event.tnm_categories.clinical_categories.clinical_t": "clinical t"}

    clin_description_original = clin_description

    if source == "broad":
        old_col = "patient.stage_event.tnm_categories.pathologic_categories.pathologic_n"
        print("Using broad labels.")
        nodal_status = pd.read_csv(INPUT_PATH + "nodal_status_PRAD_broad_processed.txt", sep="\t", index_col=0)
        nodal_status = nodal_status.applymap(lambda s: s.lower() if isinstance(s, str) else s)

        clin_description = pd.merge(clin_description, nodal_status, left_index=True, right_index=True)
        clin_description = clin_description.drop(columns=old_col)
        rename_dict["ajcc_pathologic_n"] = "pathologic n"

    clin_description = clin_description.rename(columns=rename_dict)

    rows_to_process = ["age", "psa", "n_lymph_nodes", "Gleason Score (GS)", "GS primary", "GS secondary", "pathologic t"]
    process_way = ["float", "float", "float", "categorical", "categorical", "categorical", "categorical"]

    clin_description = clin_description.loc[clin_description["pathologic n"].isin({"n0", "n1"})]
    groups = clin_description["pathologic n"].str.upper().values

    # for the summary in the paper
    clin_description['pathologic t'] = ["p" + s.upper()[:2] if str(s) != "nan" else s for s in clin_description['pathologic t']]
    clin_description["Gleason Score (GS)"] = ["< 7" if int(s) < 7 else "= 7" if int(s) == 7 else "> 7" for s in clin_description["Gleason Score (GS)"]]
    clin_description["GS primary"] = ["< 4" if int(s) < 4 else "> 3" if int(s) > 3 else s for s in clin_description["GS primary"]]
    clin_description["GS secondary"] = ["< 4" if int(s) < 4 else "> 3" if int(s) > 3 else s for s in clin_description["GS secondary"]]

    row = 'pathologic t'
    process_row_categorical(clin_description[row], groups)

    row = 'age'
    process_row_float(clin_description[row], groups)

    summaries = []
    for row, mode in zip(rows_to_process, process_way):
        df_ = process_row(clin_description[row], groups, mode=mode)

        summaries.append(df_)

    summary_table = pd.concat(summaries, axis=0)
    summary_table.to_csv(FINAL_RESULTS_PATH + "summary_table.csv")
