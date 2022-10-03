import warnings
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact, mannwhitneyu, rankdata, f_oneway, wilcoxon
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score
from utils.read_in_utils import check_id


def pair_wise_mwu_per_gene(gene_exp_df, normals="AN", other_groups=("PT", "LN"), include_fdr=False):
    tissue_types = get_tissue_types(gene_exp_df)
    odict = {"P-value %s > %s" % (normals, group): [] for group in other_groups}
    odict["Gene"] = gene_exp_df.index.values

    tag = "|".join([normals] + list(other_groups))
    odict["Ranks: %s" % tag] = []
    #odict["%s<%s"%(normals, other_group)] = []

    tissue_specific_dfs = {group: gene_exp_df.loc[:, tissue_types == group].values for group in other_groups}
    normal_df = gene_exp_df.loc[:, tissue_types == normals].values

    for i in range(gene_exp_df.shape[0]):
        medians = [np.median(normal_df[i, :])]
        for group in other_groups:
            try:
                D, pval = mannwhitneyu(normal_df[i, :], tissue_specific_dfs[group][i, :])

            except ValueError:
                D = np.NAN
                pval = 1

            odict["P-value %s > %s" % (normals, group)].append(pval)

            medians += [np.median(tissue_specific_dfs[group][i, :])]

        odict["Ranks: %s" % tag].append("|".join(rankdata(medians).astype(int).astype(str)))
        #odict["LN<AN"].append(1 * (medians[-1] < medians[0]))

    if include_fdr:
        pval_tags = [s for s in odict.keys() if s[:5] == "P-val"]

        for tag in pval_tags:
            _, odict[tag.replace("P-value", "FDR")], _, _ = multipletests(odict[tag], method='fdr_bh')

    return pd.DataFrame(odict).set_index("Gene")


def fisherExact(v, y, th=1e-15, alternative='two-sided', genes=None, gene2int=None):
    table = np.array([[np.sum((v < th) & (y == 0)), np.sum((v < th) & (y != 0))],
                      [np.sum((v >= th) & (y == 0)), np.sum((v >= th) & (y != 0))]])
    return fisher_exact(table, alternative=alternative)[1]


def find_enriched_clusters(cluster_labels, labels, alternative='two-sided'):

    cluster_labels, labels = np.array(cluster_labels), np.array(labels)
    uniq_labels = np.unique(labels)
    uniq_clusters = np.unique(cluster_labels)

    overlap_mat = np.zeros((len(uniq_labels), len(uniq_clusters)))
    pval_mat = np.zeros((len(uniq_labels), len(uniq_clusters)))

    for r, label in enumerate(uniq_labels):

        label_mask = labels == label

        for c, cluster_label in enumerate(uniq_clusters):
            cluster_mask = cluster_labels == cluster_label

            overlap_mat[r, c] = np.sum(cluster_mask & label_mask)
            pval_mat[r, c] = fisherExact(label_mask, cluster_mask, alternative=alternative)

    return pd.DataFrame(overlap_mat, index=uniq_labels, columns=uniq_clusters), \
           pd.DataFrame(pval_mat, index=uniq_labels, columns=uniq_clusters)


def convert_pval_df_to_table(pval_df, thresh=0.05):

    r, c = np.where(pval_df.values < thresh)

    return pd.DataFrame({"Cluster": pval_df.index.values[r],
                         "Hallmark": pval_df.columns.values[c],
                         "P-value": pval_df.values[(r, c)]})


def differential_testing_seeding_nonseeding(exp_df, seeding_clones, non_seeding_clones):
    seeding_mask = [g in seeding_clones for g in exp_df.columns]
    not_seeding_mask = [g in non_seeding_clones for g in exp_df.columns]

    seeding_exp_mat, non_seeding_exp_mat = exp_df.loc[:, seeding_mask].values, \
                                           exp_df.loc[:, not_seeding_mask].values

    odict = {"S>NS": [], "pval": []}
    for i in range(exp_df.shape[0]):
        v1, v2 = seeding_exp_mat[i, :], non_seeding_exp_mat[i, :]
        try:
            D, pval = mannwhitneyu(v1, v2)

        except ValueError:
            D = np.NAN
            pval = 1

        odict["S>NS"].append(np.median(v1) > np.median(v2))
        odict["pval"].append(pval)

    pval_df = pd.DataFrame(odict, index=exp_df.index).sort_values(by="pval")

    return pval_df


def anova_per_signature_df(signature_df):
    unique_groups = np.unique(signature_df.iloc[:, 1])
    values_per_signature = [signature_df.loc[signature_df.iloc[:, 1] == group].iloc[:, 0].values
                            for group in unique_groups]

    test_result = f_oneway(*values_per_signature)

    return test_result


def pair_wise_mwu(single_signature_df):
    signature_name = single_signature_df.columns.values[0]
    unique_groups = np.unique(single_signature_df.iloc[:, 1])
    mask = single_signature_df.iloc[:, 1].values
    exp_values = single_signature_df.iloc[:, 0]
    odict = {"Signature": [],
             "Group 1": [],
             "group 2": [],
             "P-value": [],
             "Statistic": []}

    for g1 in unique_groups:
        sign_g1 = exp_values[mask == g1]
        for g2 in unique_groups:
            sign_g2 = exp_values[mask == g2]
            try:
                D, pval = mannwhitneyu(sign_g1, sign_g2, alternative="greater")

            except ValueError:
                D = np.NAN
                pval = 1

            odict["Signature"].append(signature_name)
            odict["Group 1"].append(g1)
            odict["group 2"].append(g2)
            odict["P-value"].append(pval)
            odict["Statistic"].append(D)

    return pd.DataFrame(odict)


def perform_pair_wise_testing(signature_values, groups="Tissue type"):

    odf = []
    for signature in signature_values.index.values:
        signature_df = create_single_signature_df(signature_values, signature, groups=groups)
        df_ = pair_wise_mwu(signature_df)
        odf.append(df_)

    return pd.concat(odf, axis=0).sort_values(by="P-value")


def perform_one_way_anova(signature_values, groups="Patient", add_ranks=False):

    o_dict = None

    statistics, pvals = [], []
    for signature in signature_values.index.values:
        signature_df = create_single_signature_df(signature_values, signature, groups=groups)

        statistic, pval = anova_per_signature_df(signature_df)
        pvals.append(pval)
        statistics.append(statistic)

        ranks = (-1 * signature_df.groupby(groups).median()).rank().astype(int)

        if o_dict is None:
            o_dict = {group: list(rank) for group, rank in zip(ranks.index.values, ranks.values)}
        else:
            for group, rank in zip(ranks.index.values, ranks.values):
                o_dict[group] += list(rank)

    if add_ranks:
        o_dict["Statistic"] = statistics
        o_dict["p-value"] = pvals

        return pd.DataFrame(o_dict, index=signature_values.index).sort_values(by="p-value")

    else:
        return pd.DataFrame({"Statistic": statistics,
                            "p-value": pvals}, index=signature_values.index).sort_values(by='p-value')


def create_single_signature_df(signature_values, signature,  groups="Patient"):
    if groups == "Patient":
        patient_ids = [int(check_id([id_ for id_ in s.split("_") if "ID" in id_][0])[2:])
                       for s in signature_values.columns.values]

        data_dict = {signature: signature_values.loc[signature].values,
                     groups: patient_ids}

    elif groups == "Tissue type":

        tissue_types = ['PT' if 'RP' in s.split("_")[-1] else 'LN' if 'LN' in s.split("_")[-1] else 'AN'
                        for s in signature_values.columns.values]

        data_dict = {signature: signature_values.loc[signature].values,
                     groups: tissue_types}

    else:
        IOError("Groups not understood, possible values are <Patient> and <Tissue type>")

    signature_df = pd.DataFrame(data_dict,
                                index=signature_values.columns.values)

    return signature_df


def generate_boxplots(signature_values, groups="Patient"):

    for signature in signature_values.index.values:
        plt.figure()
        signature_df = create_single_signature_df(signature_values, signature, groups=groups)

        sns.boxplot(data=signature_df, x=groups, y=signature)


def get_tissue_types(data_df):
    tissue_types = ['PT' if 'RP' in s.split("_")[-1] else 'LN' if 'LN' in s.split("_")[-1] else 'AN'
                    for s in data_df.columns.values]
    return np.array(tissue_types)


def mwu_between_features(df1, df2, include_fdr=True, stat_func=mannwhitneyu):

    common_features = np.intersect1d(df1.columns.values, df2.columns.values)
    odict = {"Median_1": [], "Median_2": [], "pval": []}
    idxs = []

    medians1, medians2 = df1.median(axis=0), df2.median(axis=0)

    for common_feature in common_features:
        try:
            D, pval = stat_func(df1[common_feature].values, df2[common_feature].values)
            odict["Median_1"].append(medians1[common_feature])
            odict["Median_2"].append(medians2[common_feature])
            odict["pval"].append(pval)
            idxs.append(common_feature)

        except ValueError:
            warnings.warn("All numbers are identical for: %s." % common_feature)

    oseries = pd.DataFrame(odict, index=idxs)

    if include_fdr:

        _,  oseries["FDR"], _, _ = multipletests(oseries.pval.values.flatten(), method='fdr_bh')

    return oseries.sort_values(by="pval")


def calc_auc(v1, v2):
    v1 = np.asarray(v1).flatten()
    v2 = np.asarray(v2).flatten()

    labels = np.array([1] * len(v1) + [0] * len(v2))

    vs = np.hstack((v1, v2))

    auc = roc_auc_score(labels, vs)

    return auc


def mwu(v, y, alternative="two-sided"):

    uniq_vals = np.unique(y)

    assert len(uniq_vals) == 2, "There must be exactly two values in y to perform mwu!"

    try:
        U, pval = mannwhitneyu(v[y == uniq_vals[0]], v[y == uniq_vals[1]], alternative=alternative)

    except ValueError:
        warnings.warn("MWU Failed ...")
        U, pval = np.nan, np.nan

    return U, pval


def wilcox(v, y, alternative="two-sided"):

    uniq_vals = np.unique(y)

    assert len(uniq_vals) == 2, "There must be exactly two values in y to perform mwu!"

    try:
        U, pval = wilcoxon(v[y == uniq_vals[0]], v[y == uniq_vals[1]])

    except ValueError:
        warnings.warn("MWU Failed ...")
        U, pval = np.nan, np.nan

    return U, pval


def poisson_binom_pmf(probs):
    pmf = np.zeros(len(probs) + 1)

    pmf[0] = 1. - probs[0]
    pmf[1] = probs[0]

    for prob in probs[1:]:
        pmf = ((1-prob) * pmf + np.roll(pmf, 1) * prob)

    return pmf