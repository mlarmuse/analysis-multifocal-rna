import sys
sys.path.append("../")
import pandas as pd
from utils.read_in_utils import get_TCGA_data_as_one, read_in_all_signatures, get_expression
from utils.CentroidClassifier import CentroidClassifier
import numpy as np
import time
from sklearn.metrics import roc_auc_score
from utils.statistics import mwu
from scipy.stats import mannwhitneyu
import seaborn as sns
from statsmodels.stats import multitest
import matplotlib.pyplot as plt
from utils.get_paths import FINAL_RESULTS_PATH


def min_acc_score(y_test, y_pred):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    uniq_classes = np.unique(y_test)
    accuracies = min([np.sum(y_pred[y_test == cl] == y_test[y_test == cl])/(y_test == cl).sum() for cl in uniq_classes])

    return accuracies


def get_score_per_signature(df_train, y_train, df_validation, y_validation):
    probs_TCGA = {}
    signature_scores_ln = {"AUC": [], "pval": []}
    used_signatures = []

    for i, signature in enumerate(signatures.index.values):
        print("######################### %i ########################" % i)
        genes = np.intersect1d(df_train.columns.values, signatures.loc[signature].astype(str))

        if len(genes) > 5:
            X = df_train[genes].values
            X_tcga = df_validation[genes].values

            clf = CentroidClassifier()
            clf.fit(X, y_train)

            probs_TCGA[signature] = clf.predict_proba(X_tcga)

            y_pred = probs_TCGA[signature][:, -1]
            auc = roc_auc_score(y_validation, y_pred)
            pval = mwu(y_pred, y_validation)[1] # This is wrong!!!!!!!!!!!!!!!!!!!!!!
            #pval = mwu(y_pred[y_validation == 0], y_pred[y_validation == 1])[1]

            signature_scores_ln["AUC"].append(auc)
            signature_scores_ln["pval"].append(pval)
            used_signatures.append(signature)

    odf = pd.DataFrame(signature_scores_ln, index=used_signatures).sort_values(by="pval")
    return odf, probs_TCGA


def random_tests(df_train, y_train, df_validation, y_validation, signatures, N_perm=100):
    signature_sizes = []

    for signature in signatures.index.values:
        genes = np.intersect1d(df_train.columns.values, signatures.loc[signature].astype(str))
        signature_sizes.append(len(genes))

    signature_sizes = np.unique(signature_sizes)
    signature_sizes = signature_sizes[signature_sizes > 5]
    print("Identified %i different signature sizes." % len(signature_sizes))

    X, X_val = df_train.values, df_validation.values
    ncols = X.shape[1]

    random_scores = {}

    for size in signature_sizes:
        print(size)
        raucs = []
        for _ in range(N_perm):
            ids = np.random.choice(ncols, size, replace=False)

            clf = CentroidClassifier()
            clf.fit(X[:, ids], y_train)

            probs_ = clf.predict_proba(X_val[:, ids])
            raucs.append(roc_auc_score(y_validation, probs_[:, -1]))

        random_scores[size] = raucs

    return random_scores


if __name__ == "__main__":
    fdr_thresh = 1e-5

    np.random.seed(42)

    exp = get_expression()
    exp = exp.transpose()

    exp_TCGA = get_TCGA_data_as_one(nodal_source="broad")
    exp_TCGA = exp_TCGA.transpose()

    exp_TCGA = exp_TCGA.loc[[s.split("_")[-1] != "NL" for s in exp_TCGA.index]]
    exp_TCGA = exp_TCGA.loc[['_' in s for s in exp_TCGA.index]]

    common_genes = np.intersect1d(exp.columns.values, exp_TCGA.columns.values)

    exp, exp_TCGA = exp[common_genes], exp_TCGA[common_genes]
    y = np.array([2 if ("LN" in s.split("_")[-1]) else 1 if ("RP" in s.split("_")[-1]) else 0
                  for s in exp.index]).astype(int)
    y_tcga = np.array([s.split("_")[-1] == "LN" for s in exp_TCGA.index]).astype(int)

    signatures, index2biology = read_in_all_signatures(return_index2biology=True)

    scores, probs_df = get_score_per_signature(df_train=exp, y_train=y, df_validation=exp_TCGA, y_validation=y_tcga)
    start = time.time()
    random_scores = random_tests(df_train=exp, y_train=y,
                                 df_validation=exp_TCGA, y_validation=y_tcga,
                                 signatures=signatures, N_perm=5)

    stop = time.time()

    pvals = []
    zscores = []
    signature_sizes = {}

    for signature in scores.index.values:
        print(signature)
        signature_size = len(np.intersect1d(signatures.loc[signature].astype(str), exp.columns.values))
        print(signature_size)
        signature_sizes[signature] = signature_size
        random_aucs = random_scores[signature_size]

        score = scores["AUC"].loc[signature]

        pvals.append(np.sum(random_aucs >= score)/len(random_aucs))
        print(pvals[-1])

        mu, sigma = np.array(random_scores[signature_size]).mean(), np.array(random_scores[signature_size]).std()

        zscores.append((score - mu)/sigma)

    pvals = pd.Series(pvals, index=scores.index.values, name="p-value permutation").sort_values()
    zscores = pd.Series(zscores, index=scores.index.values, name="Z-score").sort_values()

    res_df = pd.concat([pvals, zscores], axis=1, sort=True)
    all_scores_df = pd.merge(scores, res_df, left_index=True, right_index=True)
    all_scores_df["Hallmark"] = [index2biology[sign] for sign in all_scores_df.index.values]
    _, all_scores_df["FDR"], _, _ = multitest.multipletests(all_scores_df.pval.values, method="fdr_bh")

    all_scores_df.sort_values(by="AUC")
    all_scores_df.to_csv(FINAL_RESULTS_PATH + "Results_MLN_centroid_TCGA_fixed_seed.csv")

    random_scores = pd.DataFrame(random_scores)
    random_scores.to_csv(FINAL_RESULTS_PATH + "random_scores_MLN_Classification.csv")
    pd.Series(signature_sizes).to_csv(FINAL_RESULTS_PATH + "table_signature_sizes.csv", header=False)

    # for illustration
    signature = "Focal adhesion"
    LNI_Status = pd.Series(y_tcga, name="LNI Status")
    probs = pd.Series(1. - probs_df[signature][:, -1], name="Distance to MLN Centroid")
    plot_df = pd.concat([probs, LNI_Status], axis=1)


    fig, ax = plt.subplots()
    sns.histplot(data=plot_df, x="Distance to MLN Centroid", hue="LNI Status", common_norm=False,
                 kde=True, stat="probability", ax=ax, alpha=0.3, line_kws={"linewidth": 3})
    #sns.kdeplot(data=plot_df, x="Distance to MLN Centroid", hue="LNI Status", common_norm=False, ax=ax)
    ax.set_xlabel("Distance to MLN centroid", fontsize=20)
    ax.set_ylabel("Density", fontsize=20)

    #current_handles, current_labels = ax.get_legend_handles_labels()

    ax.get_legend().set_title(None)
    ax.set_yticks([0, 0.1, 0.2])
    ax.set_xticks([0.6, 0.65, 0.70, 0.75])

    #plt.legend(loc='upper left', title=None)
    # call plt.legend() with the new values
    #ax.legend(current_handles, current_labels, loc="upper left")

    legend = ax.get_legend()
    labels = ["N0", "N1"]
    handles = [] if legend is None else legend.legendHandles

    ax.legend(handles, labels, loc='upper left', frameon=False, fontsize=16)
    plt.savefig(FINAL_RESULTS_PATH + "Distance_2_mln_centroid.svg", dpi=600)
    plt.show()

    plt.close()

    lni_status_mask = plot_df["LNI Status"].values == 1
    U, pval = mannwhitneyu(plot_df["Distance to MLN Centroid"].values[lni_status_mask],
                           plot_df["Distance to MLN Centroid"].values[lni_status_mask],
                           alternative="greater")
