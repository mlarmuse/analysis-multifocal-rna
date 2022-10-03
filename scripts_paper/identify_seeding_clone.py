import sys
sys.path.append("../")

import pandas as pd
from utils.read_in_utils import read_in_all_signatures, get_expression
from utils.CentroidClassifier import CentroidClassifier
import numpy as np
from utils.CentroidClassifier import spearman_dist_mat
from utils.statistics import poisson_binom_pmf
from scipy.stats import mannwhitneyu
from utils.read_in_utils import INPUT_PATH, get_confirmed_patients
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import hypergeom
from utils.plots import sort_ids_by_patient_sample_type_number
from statsmodels.stats.multitest import multipletests
from utils.get_paths import FINAL_RESULTS_PATH


def get_inter_intra_dists(exp, input_genes=None, source_tissue="RP", target_tissue="MLN"):
    identical_tissue = source_tissue == target_tissue

    exp = exp.copy(deep=True)

    if input_genes is not None:
        common_genes = np.intersect1d(exp.columns.values, input_genes)
        exp = exp[common_genes]

    rp_exp = exp.loc[[s for s in exp.index if source_tissue in s]]
    mln_exp = exp.loc[[s for s in exp.index if target_tissue in s]]

    dists = spearman_dist_mat(rp_exp.values, mln_exp.values)
    dists = pd.DataFrame(dists, index=rp_exp.index, columns=mln_exp.index)

    pat_mask = np.array([s.split("_")[1] for s in dists.index])[..., None] == \
               np.array([s.split("_")[1] for s in dists.columns])[None, ...]

    if identical_tissue:
        pat_mask[np.triu_indices_from(pat_mask)] = False
        source_tissue = source_tissue + "1"
        target_tissue = source_tissue + "2"

    intra_pat_dists = dists.values[pat_mask]
    r, c = np.where(pat_mask)
    intra_patient_df = pd.DataFrame({source_tissue: dists.index.values[r],
                                     target_tissue: dists.columns.values[c],
                                     "Affinity": dists.values[(r, c)]})
    if identical_tissue:
        pat_mask[np.triu_indices_from(pat_mask)] = True

    inter_pat_dists = dists.values[~pat_mask]
    r, c = np.where(~pat_mask)
    inter_patient_df = pd.DataFrame({source_tissue: dists.index.values[r],
                                     target_tissue: dists.columns.values[c],
                                     "Affinity": dists.values[(r, c)]})

    U, pval = mannwhitneyu(intra_pat_dists, inter_pat_dists, alternative="greater")

    return intra_patient_df, inter_patient_df, pval


def get_probs_per_signature(exp=None, y=None, signatures=None, return_all=True):

    if exp is None:
        exp = get_expression()
        exp = exp.transpose()

    if y is None:
        y = np.array([2 if ("LN" in s.split("_")[-1]) else 1 if ("RP" in s.split("_")[-1]) else 0
                      for s in exp.index]).astype(int)

    if isinstance(signatures, list) or isinstance(signatures, np.ndarray):
        signature_df, index2biology = read_in_all_signatures(return_index2biology=True)
        common_signatures = np.intersect1d(signature_df.index.values, signatures)
        if len(common_signatures):
            raise IOError("No single signature name is known. Please check with the names obtained through"
                          " <read_in_all_signatures()>.")
        signatures = signature_df.loc[common_signatures]

    if signatures is None:
        signatures, index2biology = read_in_all_signatures(return_index2biology=True)

    probs_per_signature = {}

    for signature in signatures.index.values:

        common_genes = np.intersect1d(signatures.loc[signature].astype(str), exp.columns.values)
        if len(common_genes) > 5:
            X = exp[common_genes].values

            clf = CentroidClassifier()
            clf.fit(X, y)

            X_rp = X[y == 1]
            probs_per_signature[signature] = clf.predict_proba(X_rp)[:, -1]

    probs_per_signature = pd.DataFrame(probs_per_signature, index=exp.index.values[y == 1])

    if return_all:
        return probs_per_signature, exp, y, signatures
    else:
        return probs_per_signature


def get_centroid_per_signature(exp=None, y=None, signatures=None, return_all=True):

    if exp is None:
        exp = get_expression()
        exp = exp.transpose()

    if y is None:
        y = np.array(["MLN" if ("LN" in s.split("_")[-1]) else "RP" if ("RP" in s.split("_")[-1]) else "NL"
                      for s in exp.index])

    if isinstance(signatures, list) or isinstance(signatures, np.ndarray):
        signature_df, index2biology = read_in_all_signatures(return_index2biology=True)
        common_signatures = np.intersect1d(signature_df.index.values, signatures)
        if len(common_signatures):
            raise IOError("No single signature name is known. Please check with the names obtained through"
                          " <read_in_all_signatures()>.")
        signatures = signature_df.loc[common_signatures]

    if signatures is None:
        signatures, index2biology = read_in_all_signatures(return_index2biology=True)

    centroids_per_signature = {}

    for signature in signatures.index.values:

        common_genes = np.intersect1d(signatures.loc[signature].astype(str), exp.columns.values)

        if len(common_genes) > 5:
            X = exp[common_genes]

            clf = CentroidClassifier()
            clf.fit(X, y)

            centroids_per_signature[signature] = clf.centroid_df

    if return_all:
        return centroids_per_signature, exp, y, signatures
    else:
        return centroids_per_signature


def majority_vote_seeds(probs_per_signature, index2biology=None, vote_per_hm=False):
    """
    :param probs_per_signature: a pd DF containing samples as index and signatures as columns
    :param index2biology: a map from signature to hallmark
    :return:
    """

    if index2biology is None:
        _, index2biology = read_in_all_signatures(return_index2biology=True)

    patients = np.array([s.split("_")[1] for s in probs_per_signature.index.values])
    uniq_pats = np.unique(patients)
    binary_dfs, votes_df, seeds, vote_fractions = [], [], [], []
    non_seeds = []
    pat_index = []

    for pid in uniq_pats:
        print(pid)
        mat = probs_per_signature.loc[patients == pid]
        mat = mat == np.max(mat.values, axis=0, keepdims=True)
        binary_dfs.append(mat)

        norm_votes = mat.astype(int)/np.sum(mat.values, axis=0, keepdims=True)
        norm_votes = norm_votes.fillna(0)

        votes_df.append(norm_votes)
        votes = norm_votes.sum(axis=1)
        pat_seeds = list(votes.index.values[votes.values == votes.max()])
        seeds += pat_seeds
        non_seeds += [votes.idxmin()]

        votes_fraction = votes.sort_values(ascending=False).values
        votes_fraction = votes_fraction[0]/votes_fraction[1]

        vote_fractions.append(votes_fraction)
        pat_index.append(pid)

        if len(pat_seeds) > 1:
            vote_fractions.append(votes_fraction)
            pat_index.append(pid)

        elif len(pat_seeds) == 0:
            print("No seed found for patient %s" % pid)

        print(mat.shape[0])
        print("Fraction of votes for the best performing sample: %f" % (votes[-1]/votes.sum()))
        print("Fraction of best to second best: %f" % (votes[-1]/votes[-2]))

    binary_dfs = pd.concat(binary_dfs, axis=0)
    votes_df = pd.concat(votes_df, axis=1, sort=True)
    vote_fractions = pd.Series(vote_fractions, index=pat_index)

    return binary_dfs, votes_df, vote_fractions, seeds, non_seeds


def get_continuous_metastatic_score(probs_per_signature, index2biology=None, vote_per_hm=True):
    if index2biology is None:
        _, index2biology = read_in_all_signatures(return_index2biology=True)

    hallmarks = np.array([index2biology[s] for s in probs_per_signature.columns.values])

    if vote_per_hm:
        probs_per_sample = probs_per_signature.transpose().groupby(hallmarks).mean().mean()

    else:
        probs_per_sample = probs_per_signature.mean(axis=1)

    return probs_per_sample


def normalize_votes_per_patient(binary_vote_df):
    binary_vote_df = (binary_vote_df.copy(deep=True) > 0).astype(int)
    patients = np.array([s.split("_")[1] for s in binary_vote_df.index.values])
    uniq_pats = np.unique(patients)
    votes_df = []

    for pid in uniq_pats:
        print(pid)
        mat = binary_vote_df.loc[patients == pid]
        norm_votes = mat == np.max(mat.values, axis=1, keepdims=True)
        norm_votes = norm_votes.astype(int)/np.sum(norm_votes.values, axis=0, keepdims=True)
        norm_votes = norm_votes.fillna(0)
        votes_df.append(norm_votes)

    return pd.concat(votes_df, axis=0)


def test_normalize_votes_per_patient():
    test_df1 = pd.DataFrame({"S1": [0, 0, 1], "S2": [0, 1, 0], "S3": [1, 0, 1]}, index=["HR_ID1_RP1",
                                                                                        "HR_ID1_RP2",
                                                                                        "HR_ID1_RP3"])

    test_df2 = pd.DataFrame({"S1": [1, 0, 0, 1], "S2": [0, 1, 0, 0], "S3": [1, 0, 1, 1]}, index=["HR_ID2_RP1",
                                                                                                 "HR_ID2_RP2",
                                                                                                 "HR_ID2_RP3",
                                                                                                 "HR_ID2_RP4"])
    print(normalize_votes_per_patient(test_df1))
    print(normalize_votes_per_patient(test_df2))
    print(normalize_votes_per_patient(pd.concat([test_df1, test_df2], axis=0)))


def overrepresentation_in_seeding_clone(binary_vote_df, seeds):
    seeding_labels = np.array([s in seeds for s in binary_vote_df.index])

    counts_per_signature = (binary_vote_df > 0).astype(int).transpose().dot(seeding_labels).sort_values()

    normalized_vote_df = normalize_votes_per_patient(binary_vote_df)
    score_per_signature = normalized_vote_df.transpose().dot(seeding_labels).sort_values()

    # added: remove patients that do not have a seeding lesion, we think this is the only correct behavior:
    seeding_pats = np.array([s.split("_")[1] for s in seeds])
    mask = [s.split("_")[1] in seeding_pats for s in binary_vote_df.index.values]
    binary_vote_df = binary_vote_df.loc[mask]

    # to test the significance of these hallmarks
    samples = np.array([s.split("_")[1] for s in binary_vote_df.index.values])
    probs_per_patient = {}
    uniq_samples_with_seed = np.unique([s.split("_")[1] for s in seeds])
    print("There are %i patients for which the seed is known." % len(np.unique(uniq_samples_with_seed)))

    for sid in uniq_samples_with_seed: # TODO: extend to more than 1 seed per patient
        mask = samples == sid

        if mask.sum() > 0:
            mat = binary_vote_df.loc[samples == sid]

            pat_probs = mat.sum(axis=0)/mat.shape[0]
            probs_per_patient[sid] = pat_probs

    probs_per_patient = pd.DataFrame(probs_per_patient)

    pvals, pvals_l = [], []

    for sign in counts_per_signature.index:
        score_ = counts_per_signature.loc[sign]
        pmf = poisson_binom_pmf(probs_per_patient.loc[sign])
        pvals.append(pmf[score_:].sum())
        pvals_l.append(pmf[:(score_+1)].sum())

    counts_per_signature = counts_per_signature.to_frame()
    counts_per_signature["Vote score"] = score_per_signature
    counts_per_signature["pval"] = pvals
    counts_per_signature["l-pval"] = pvals_l
    counts_per_signature.columns = ["# Correct votes", "Vote score", "p-value", "left p-value"]
    counts_per_signature = counts_per_signature.sort_values(by="p-value", ascending=True)

    return counts_per_signature, probs_per_patient


if __name__ == "__main__":
    fdr_thresh = 1e-2
    vote_per_hm = True
    np.random.seed(42)
    confirmed_patients = True
    tag = ""

    # Read in the Z-scores per signature, based on the TCGA primaries
    zscores = pd.read_csv(FINAL_RESULTS_PATH + "Results_MLN_centroid_TCGA_fixed_seed.csv", index_col=0)
    predictive_signatures = zscores.loc[zscores["FDR"] < fdr_thresh].index
    non_predictive_signatures = np.setdiff1d(zscores.index.values, predictive_signatures)

    # get the MLN probs per signature
    probs_per_signature, exp, y, signatures = get_probs_per_signature()

    # revert for Mitochondrial
    probs_per_signature["Mitochondrial: Complex IV"] = 0.66 - probs_per_signature["Mitochondrial: Complex IV"]

    # simple appraoch: majority vote
    binary_dfs, vote_df, _, seeds2, nonseeds2 = majority_vote_seeds(probs_per_signature[zscores.index.values])

    _, votes_df, vote_fractions, seeds, non_seeds = majority_vote_seeds(probs_per_signature[predictive_signatures], vote_per_hm=False)

    pd.DataFrame({"Seeds": seeds, "Non-Seeds": non_seeds}).to_csv(FINAL_RESULTS_PATH + "seeds_nonseeds.csv")
    binary_dfs.to_csv(FINAL_RESULTS_PATH + "binary_df.csv", index=True)

    # binary_dfs.to_csv("results/figures_table_paper/binary_df.csv", index=True)

    # continuous scoring based on model performance
    score_per_sample = get_continuous_metastatic_score(probs_per_signature[zscores.index.values])
    scores_df = pd.DataFrame({"Votes": binary_dfs[predictive_signatures].sum(axis=1),
                              "Score": score_per_sample})

    scores_df.to_csv(FINAL_RESULTS_PATH + "different_seed_scoring_schemes_signatures.csv", index=True)

    if confirmed_patients:
        tag = "_confirmed_patients"
        confirmed_patients = get_confirmed_patients()
        probs_per_signature = probs_per_signature.loc[[s.split("_")[1] in confirmed_patients
                                                       for s in probs_per_signature.index]]

        binary_dfs = binary_dfs.loc[[s.split("_")[1] in confirmed_patients
                                     for s in binary_dfs.index]]

    score_per_sample = get_continuous_metastatic_score(probs_per_signature[zscores.index.values])

    scores_df = pd.DataFrame({"Votes": binary_dfs[predictive_signatures].sum(axis=1),
                              "Score": score_per_sample})
    scores_df.to_csv(FINAL_RESULTS_PATH + "different_seed_scoring_schemes_signatures%s.csv" % tag, index=True)
    # check which signatures are also special in the seeding clones
    scores_per_signature, _ = overrepresentation_in_seeding_clone(binary_dfs, seeds)
    scores_per_signature = scores_per_signature.sort_values(by="p-value")
    _, scores_per_signature["q-value"], _, _ = multipletests(scores_per_signature["p-value"], method="fdr_bh")
    scores_per_signature.to_csv(FINAL_RESULTS_PATH + "scores_seeding_signatures%s.csv" % tag,
                                index=True)


    # link with pam50 data
    pam50_scores = pd.read_csv(INPUT_PATH + "pam50_TPM.csv", index_col=0)

    pam50_scores = pam50_scores.loc[["RP" in s.split("_")[-1] for s in pam50_scores.index]]
    pam50_scores = pam50_scores[["Basal", "LumA", "LumB"]]/pam50_scores[["Basal", "LumA", "LumB"]].values.sum(axis=1, keepdims=True)

    pam50_labels = pam50_scores.idxmax(axis=1).to_frame(name="pam50_label")
    pam50_labels["seeding"] = pam50_labels.index.isin(seeds)

    idx = sort_ids_by_patient_sample_type_number(pam50_labels.index.values)
    pam50_labels = pam50_labels.loc[idx]

    pd.crosstab(pam50_labels.seeding, pam50_labels.pam50_label)

    strict_seeds = np.array(seeds)[vote_fractions > 1.]
    pam50_labels["strict_seeding"] = pam50_labels.index.isin(strict_seeds)

    pd.crosstab(pam50_labels.strict_seeding, pam50_labels.pam50_label)
    patients = np.array([s.split("_")[1] for s in pam50_labels.index.values])
    uniq_patients = np.unique(patients)

    probs_per_patient_pam50 = {}
    subtype, counter = "LumB", 0

    for pat in uniq_patients:

        pam50_labels_pat = pam50_labels.loc[patients == pat]

        n_lumb = (pam50_labels_pat.pam50_label == subtype).sum()
        n_seeding = pam50_labels_pat.seeding.sum()

        counter += (1 * pam50_labels_pat.pam50_label == subtype).dot(1 * pam50_labels_pat.seeding) > 0

        [M, n, N] = [pam50_labels_pat.shape[0], n_lumb, n_seeding]
        rv = hypergeom(M, n, N)

        probs_per_patient_pam50[pat] = 1. - rv.pmf(0)

    probs_per_patient_pam50 = pd.Series(probs_per_patient_pam50)
    probs_per_patient_pam50["ID16"] = 0.
    poisson_pmf = poisson_binom_pmf(probs_per_patient_pam50.values)

    pval = poisson_pmf[counter:].sum()

    print("P-value for subtype %s is: %f" % (subtype, pval))

    # Finally check the heterogeneity

    pval_per_signature = {}

    for signature in zscores.index.values:

        common_genes = np.intersect1d(signatures.loc[signature].astype(str), exp.columns.values)

        if len(common_genes) > 5:
            _, _, pval = get_inter_intra_dists(exp, signatures.loc[signature].astype(str),
                                               target_tissue="RP",
                                               source_tissue="RP")
            pval_per_signature[signature] = pval

    pvals_series = pd.Series(pval_per_signature).sort_values()

    #
    signature = "Focal adhesion"

    intra_pat_dists, inter_pat_dists, pval = get_inter_intra_dists(exp, signatures.loc[signature].astype(str))

    plot_df = {"RP - MLN distance": np.concatenate((intra_pat_dists.Affinity.values,
                                                    inter_pat_dists.Affinity.values)),
               "Same patient": [1] * len(intra_pat_dists) + [0] * len(inter_pat_dists)}
    plot_df = pd.DataFrame(plot_df)

    plt.figure()
    sns.kdeplot(data=plot_df, x="RP - MLN distance", hue="Same patient", common_norm=False)
    plt.show()
    plt.title(signature)