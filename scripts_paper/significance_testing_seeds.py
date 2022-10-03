import sys
sys.path.append("../")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from identify_seeding_clone import get_probs_per_signature
from utils.read_in_utils import get_expression, get_signature_genes, get_TCGA_data_as_one, get_signature_sizes, get_signature_voting_seeds, get_index_lesions, get_confirmed_patients
from utils.CentroidClassifier import CentroidClassifier, spearman_dist_mat
from utils.get_paths import FINAL_RESULTS_PATH
from utils.statistics import mwu
from sklearn.metrics import roc_auc_score
from utils.plots import get_patient2color, sort_ids_by_patient_sample_type_number
from statsmodels.stats.multitest import multipletests
from joblib import delayed, Parallel


def get_random_seeds(sample_list, N_sets=1000):
    sample_list = np.asarray(sample_list)
    sample_list_int = np.arange(len(sample_list))
    patients = np.asarray([s.split("_")[1] for s in sample_list])

    uniq_patients = np.unique(patients)

    rsamples = [np.random.choice(sample_list_int[patients == pat], N_sets, replace=True) for pat in uniq_patients]
    rsamples = np.array(rsamples).T

    r_labels = []

    for rs in rsamples:

        zs = np.zeros(len(sample_list))
        zs[rs] = 1
        r_labels.append(zs)

    return np.array(r_labels).T


def get_random_probs_per_signature(signature_size, exp=None, y=None, only_cancer_genes=True,
                                   include_TCGA=False, N_perm=1000, random_state=42):
    np.random.seed(random_state)
    if exp is None:
        exp = get_expression()
        exp = exp.transpose()

    else:
        exp = exp.copy(deep=True)

    if y is None:
        y = np.array([2 if ("LN" in s.split("_")[-1]) else 1 if ("RP" in s.split("_")[-1]) else 0
                      for s in exp.index]).astype(int)

    if only_cancer_genes:
        cancer_genes = get_signature_genes(collection_name="all")
        cancer_genes = np.intersect1d(cancer_genes, exp.columns.values)
        exp = exp[cancer_genes]

    if include_TCGA:
        exp_TCGA = get_TCGA_data_as_one(nodal_source="broad")
        exp_TCGA = exp_TCGA.transpose()

        exp_TCGA = exp_TCGA.loc[[s.split("_")[-1] != "NL" for s in exp_TCGA.index]]
        exp_TCGA = exp_TCGA.loc[['_' in s for s in exp_TCGA.index]]

        common_genes = np.intersect1d(exp.columns.values, exp_TCGA.columns.values)

        exp, exp_TCGA = exp[common_genes], exp_TCGA[common_genes]
        y_tcga = np.array([s.split("_")[-1] == "LN" for s in exp_TCGA.index]).astype(int)
        X_tcga = exp_TCGA.values
        average_probs = np.zeros((X_tcga.shape[0]))

    mat = exp.values

    random_sets = [np.random.choice(mat.shape[1], size=(signature_size, ), replace=False) for _ in range(N_perm)]
    random_probs = np.zeros(((y == 1).sum(), N_perm))
    random_pvals_tcga = {}

    for i, random_set in enumerate(random_sets):
        X = mat[:, random_set]

        clf = CentroidClassifier()
        clf.fit(X, y)

        X_rp = X[y == 1]
        random_probs[:, i] = clf.predict_proba(X_rp)[:, -1]

        if include_TCGA:
            probs_TCGA = clf.predict_proba(X_tcga[:, random_set])[:, -1]

            pval = mwu(probs_TCGA, y_tcga, alternative="less")[1]
            random_pvals_tcga[i] = [pval] + [roc_auc_score(y_tcga, probs_TCGA)]

            if pval < 1e-5:
                average_probs += probs_TCGA

    random_probs = pd.DataFrame(random_probs, index=exp.index.values[y == 1])

    if include_TCGA:

        auroc = roc_auc_score(y_tcga, average_probs)
        print("The AUC averaged across all random sets is: %f" % auroc)

        random_pvals_tcga = pd.DataFrame(random_pvals_tcga, index=["P_value", "AUC"]).transpose()
        _, random_pvals_tcga["q_value"], _, _ = multipletests(random_pvals_tcga["P_value"], method="fdr_bh")

        return random_probs, random_pvals_tcga

    else:
        return random_probs


def majority_vote_seeds_random(probs_per_signature):
    """

    :param probs_per_signature: a pd DF containing samples as index and signatures as columns
    :return:
    """
    patients = np.array([s.split("_")[1] for s in probs_per_signature.index.values])
    uniq_pats = np.unique(patients)
    binary_dfs, vote_df = [], []
    seeds = []

    for pid in uniq_pats:
        mat = probs_per_signature.loc[patients == pid]
        mat = mat == np.max(mat.values, axis=0, keepdims=True)

        binary_dfs.append(mat)
        mat = mat/np.sum(mat.values, axis=0, keepdims=True)
        vote_df.append(mat)

        votes = mat.sum(axis=1)
        pat_seeds = list(votes.index.values[votes.values == votes.max()])
        seeds += pat_seeds

    binary_dfs = pd.concat(binary_dfs, axis=0)
    vote_df = pd.concat(vote_df, axis=0)

    return binary_dfs, seeds, vote_df


def plot_counts(plot_df):
    sorted_rows = sort_ids_by_patient_sample_type_number(plot_df.index.values)
    plot_df = plot_df.loc[sorted_rows]

    patient2color = get_patient2color()
    row_colors = pd.Series({pat: patient2color[pat.split("_")[1]] for pat in plot_df.index.values})

    patients = np.array([s.split("_")[1] for s in plot_df.index.values])
    uniq_pats = pd.unique(patients)
    xs = []
    currx = 0
    total = len(patients)

    print(uniq_pats)

    for pat in uniq_pats:
        currx += (patients == pat).sum() / total / 2
        xs.append(currx)
        currx += (patients == pat).sum() / total / 2

    fig, barplot_ax = plt.subplots()
    sums_per_pat = (plot_df > 0).sum(axis=1)
    barwidth = 0.8 * 1 / plot_df.shape[0]
    ys = np.arange(plot_df.shape[0]) / plot_df.shape[0]
    barplot_ax.bar(ys[::-1] + barwidth / 2, sums_per_pat.values, color=row_colors.to_list(), width=barwidth)
    barplot_ax.set_xlim([0, 1])
    barplot_ax.set_xticks(xs)
    barplot_ax.set_xticklabels(uniq_pats, rotation=45)

    barplot_ax.set_yticks([])

    barplot_ax.spines["left"].set_visible(False)
    barplot_ax.spines["top"].set_visible(False)
    barplot_ax.spines["right"].set_visible(False)
    barplot_ax.spines["bottom"].set_visible(False)

    barplot_ax.set_xlabel("Patient ID")
    barplot_ax.set_xlabel("Count")
    plt.show()


def get_vote_df_random_seeds(signature_size=10,
                             exp=None, source_type="RP", target_type="LN", random_state=42,
                             only_cancer_genes=True, N_perm=1000):

    np.random.seed(random_state)
    if exp is None:
        exp = get_expression()
        exp = exp.transpose()

    else:
        exp = exp.copy(deep=True)

    if only_cancer_genes:
        cancer_genes = get_signature_genes(collection_name="all")
        cancer_genes = np.intersect1d(cancer_genes, exp.columns.values)
        exp = exp[cancer_genes]

    patients = np.array([s.split("_")[1] for s in exp.index])
    all_votes = {}

    sources = exp.loc[[source_type in s for s in exp.index]]
    targets = exp.loc[[target_type in s for s in exp.index]]

    spatients = np.array([s.split("_")[1] for s in sources.index])
    tpatients = np.array([s.split("_")[1] for s in targets.index])

    ssamples = sources.index.values
    tsamples = targets.index.values

    sources, targets = sources.values, targets.values
    random_sets = [np.random.choice(sources.shape[1], size=(signature_size, ), replace=False) for _ in range(N_perm)]

    for pat in patients:
        tmask = tpatients == pat
        smask = spatients == pat

        pat_sources = sources[smask]
        pat_targets = targets[tmask]

        if (pat_sources.shape[0] > 0) and (pat_targets.shape[0] > 0):
            pat_votes = np.zeros((pat_sources.shape[0], pat_targets.shape[0]))
            for i, random_set in enumerate(random_sets):

                pat_sources_i = pat_sources[:, random_set]
                pat_targets_i = pat_targets[:, random_set]

                spearman_mat = spearman_dist_mat(pat_sources_i, pat_targets_i)

                votes_ = spearman_mat == np.max(spearman_mat, axis=0, keepdims=True)
                votes_ = votes_/np.sum(votes_, axis=0)
                pat_votes += votes_

            all_votes[pat] = pd.DataFrame(pat_votes,
                                          columns=tsamples[tmask],
                                          index=ssamples[smask])

    return all_votes


def random_testing_pipeline(signature_size, exp=None, y=None, N_perm=10_000, only_cancer_genes=True,
                            seeds=None, use_rankscores=False):
    print("Started random testing for signatures of size: %i" %signature_size)

    random_probs = get_random_probs_per_signature(signature_size=signature_size,
                                                  exp=exp,
                                                  y=y,
                                                  only_cancer_genes=only_cancer_genes,
                                                  include_TCGA=False,
                                                  N_perm=N_perm)

    if seeds is None:
        binary_vote_df, seeds, vote_df = majority_vote_seeds_random(random_probs)

    else:
        binary_vote_df, _, vote_df = majority_vote_seeds_random(random_probs)

    seeding_labels = np.array([s in seeds for s in binary_vote_df.index])
    scores_per_signature = vote_df.values.transpose().dot(seeding_labels)

    return pd.Series(scores_per_signature, name=signature_size)


def parallel_testing_pipeline(signature_sizes, exp=None, y=None, N_perm=10_000, only_cancer_genes=True, n_jobs=1,
                              seeds=None, use_rankscores=False):

    if n_jobs == 1:
        rvotes = [random_testing_pipeline(signature_size=ss, exp=exp, y=y, seeds=seeds, use_rankscores=use_rankscores,
                                          N_perm=N_perm, only_cancer_genes=only_cancer_genes) for ss in signature_sizes]
    else:
        rvotes = Parallel(n_jobs=n_jobs, backend='loky')(delayed(random_testing_pipeline)(signature_size=ss,
                                                                                          exp=exp,
                                                                                          y=y,
                                                                                          N_perm=N_perm,
                                                                                          only_cancer_genes=only_cancer_genes,
                                                                                          seeds=seeds,
                                                                                          use_rankscores=use_rankscores
                                                                                          )
                                                                  for ss in signature_sizes)

    return pd.concat(rvotes, axis=1)


if __name__ == "__main__":
    np.random.seed(43)
    confirmed_patients_only = True

    # get the MLN probs per signature
    probs_per_signature, exp, y, signatures = get_probs_per_signature()
    sign2size, uniq_sizes = get_signature_sizes(exp, signatures)
    uniq_sizes = np.array(list(uniq_sizes))

    tag = ""
    if confirmed_patients_only:
        confirmed_patients = get_confirmed_patients()
        exp_mask = np.array([s.split("_")[1] in confirmed_patients for s in exp.index])

        y = y[exp_mask]
        exp = exp.loc[exp_mask]
        tag = "_confirmed_patients"


    seeds = get_signature_voting_seeds()
    # signature_sizes = np.concatenate((list(uniq_sizes), np.linspace(205, 545, 69).astype(int)))
    signature_sizes = np.sort(uniq_sizes[uniq_sizes > 5])

    random_scores = parallel_testing_pipeline(signature_sizes=signature_sizes,
                                              exp=exp,
                                              y=y,
                                              N_perm=5,
                                              n_jobs=1,
                                              only_cancer_genes=False,
                                              seeds=seeds)

    random_scores.to_csv(FINAL_RESULTS_PATH + "randomscores_seeding_signature_size_all_genes%s.csv" % tag)

    index_lesions = get_index_lesions()
    random_scores_index_lesions = parallel_testing_pipeline(signature_sizes=signature_sizes,
                                                            exp=exp,
                                                            y=y,
                                                            N_perm=5,
                                                            n_jobs=1,
                                                            only_cancer_genes=False,
                                                            seeds=index_lesions.values.flatten())

    random_scores_index_lesions.to_csv(FINAL_RESULTS_PATH + "randomscores_index_lesion_signature_size_all_genes%s.csv" % tag)

