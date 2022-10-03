'''
Script to perform centroid based classification on UZ Ghent and the corresponding background testing.
'''

import sys
sys.path.append("../")

from utils.read_in_utils import get_expression
import pandas as pd
from utils.get_paths import FINAL_RESULTS_PATH
from utils.read_in_utils import read_in_all_signatures, get_signature_genes
from utils.CentroidClassifier import CentroidClassifier
import numpy as np
from joblib import Parallel, delayed
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from utils.misc import process_table
import time


start = time.time()


def get_train_test_index(exp_df, n_repeats=5, random_state=43):
    np.random.seed(random_state)
    idx = np.arange(exp_df.shape[0])
    samples = np.array([s.split("_")[1] for s in exp_df.index.values])
    uniq_samples = np.unique(samples)

    n_test_samples = np.int(len(uniq_samples) * 0.3)
    train_test_indices = []

    for _ in range(n_repeats):
        rsamples = np.random.choice(uniq_samples, n_test_samples, replace=False)

        train_test_indices.append((idx[~np.isin(samples, rsamples)], idx[np.isin(samples, rsamples)]))

    return train_test_indices


def min_acc_score(y_test, y_pred):
    y_test = np.asarray(y_test)
    y_pred = np.asarray(y_pred)
    uniq_classes = np.unique(y_test)
    accuracies = min([np.sum(y_pred[y_test == cl] == y_test[y_test == cl])/(y_test == cl).sum() for cl in uniq_classes])

    return accuracies


def get_random_AUCs(exp_df, signature, background_genes, signature_name=None, iteration=None, N_perm=1000,
                    random_state=42, train_test_indices=None):

    if train_test_indices is None:
        train_test_indices = get_train_test_index(exp_df)

    np.random.seed(random_state)
    if iteration is not None:
        print("Iteration %i" % iteration)

    aucs_ssp = []
    ngenes = len(np.intersect1d(exp_df.columns.values, signature))

    for _ in range(N_perm):
        genes = np.random.choice(background_genes, ngenes)

        if len(genes) > 5:

            y = np.array([2 if ("LN" in s.split("_")[-1]) else 1 if ("RP" in s.split("_")[-1]) else 0
                          for s in exp.index]).astype(int)

            X = exp_df[genes].values

            for train_index, test_index in train_test_indices:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf = CentroidClassifier()
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                auc = min_acc_score(y_test, preds)
                aucs_ssp.append(auc)

    return aucs_ssp, signature_name


if __name__ == "__main__":
    N_perm = 5
    fdr_threshold = 1e-5
    only_cancer_genes = False
    n_jobs = 2
    n_repeats = 2

    np.random.seed(42)

    exp = get_expression()
    exp = exp.transpose()

    signatures, index2biology = read_in_all_signatures(return_index2biology=True)
    accs_per_signature = {}

    train_test_indices = get_train_test_index(exp, n_repeats=n_repeats)
    sign2size = {}
    for i, signature in enumerate(signatures.index.values):
        print("######################### %i ########################" % i)
        genes = np.intersect1d(exp.columns.values, signatures.loc[signature].astype(str))

        if len(genes) > 5:
            y = np.array([2 if ("LN" in s.split("_")[-1]) else 1 if ("RP" in s.split("_")[-1]) else 0
                          for s in exp.index]).astype(int)

            X = exp[genes].values
            accs_ssp = []

            for train_index, test_index in train_test_indices:
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                clf = CentroidClassifier()
                clf.fit(X_train, y_train)
                preds = clf.predict(X_test)

                acc = min_acc_score(y_test, preds)
                accs_ssp.append(acc)

            sign2size[signature] = len(genes)
            accs_per_signature[signature] = accs_ssp

    accs_per_signature = pd.DataFrame(accs_per_signature)
    median_accs = accs_per_signature.median().sort_values(ascending=False)
    std_accs = accs_per_signature.std().sort_values(ascending=False)
    all_signature_genes = get_signature_genes(collection_name="all")

    if only_cancer_genes:
        background_genes = np.intersect1d(all_signature_genes, exp.columns.values)

    else:
        background_genes = np.intersect1d(all_signature_genes, exp.columns.values)

    if n_jobs == 1:
        random_aucs_per_signature = {}
        for i, signature in enumerate(accs_per_signature.columns.values):
            print("Signature %i: %s" % (i, signature))

            accs_ssp, sign_name = get_random_AUCs(exp,
                                                  signatures.loc[signature].astype(str),
                                                  background_genes,
                                                  train_test_indices=train_test_indices,
                                                  signature_name=signature)

            random_aucs_per_signature[sign_name] = accs_ssp
            print(np.mean(accs_ssp))

    else:
        aucs = Parallel(n_jobs=n_jobs)(delayed(get_random_AUCs)(exp,
                                                                signatures.loc[signature].astype(str),
                                                                background_genes,
                                                                signature_name=signature,
                                                                iteration=i,
                                                                train_test_indices=train_test_indices)
                           for i, signature in enumerate(accs_per_signature.columns.values))

        random_aucs_per_signature = {sign: values for values, sign in aucs if len(values) > 0}

    random_aucs_per_signature = pd.DataFrame(random_aucs_per_signature)

    pvals_signatures = {}

    for signature in random_aucs_per_signature.columns:
        scores = accs_per_signature[signature]
        rscores = random_aucs_per_signature[signature]
        pvals_signatures[signature] = mannwhitneyu(scores.values, rscores, alternative="greater")[1]

    n_genes = {}

    for signature in median_accs.index:
        genes = np.intersect1d(exp.columns.values, signatures.loc[signature].astype(str))
        n_genes[signature] = len(genes)

    pval_series = pd.Series(pvals_signatures).to_frame()
    pval_series.columns = ["pval"]

    pval_series["N_genes"] = [n_genes[signature] for signature in pval_series.index]
    pval_series["median_score"] = accs_per_signature.median()
    _, pval_series["FDR"], _, _ = multipletests(pval_series.pval.values.flatten(), method='fdr_bh')

    pval_series["Hallmark"] = [index2biology[t] for t in pval_series.index]
    check_df = pval_series.sort_values(by="FDR")

    otable = process_table(check_df)
    otable = otable[["N_genes", "median_score", "FDR", "Hallmark"]]
    otable = otable.rename(columns={"N_genes": "# genes",
                                    "median_score": "median AUC"})

    short_name_dict = {"Activating invasion and metastasis": "invasion",
                       "Inducing angiogenesis": "angiogenesis",
                       "Deregulation of cellular energetics": "energetics",
                       "Enabling replicative immortality": "immortality",
                       "Cancer-immunology": "immunology"
                       }

    otable["Hallmark"] = otable["Hallmark"].apply(lambda x: short_name_dict[x]
                                                            if x in short_name_dict.keys() else x)

    otable.to_csv(FINAL_RESULTS_PATH + "Classification_table_UZ_Ghent.csv")
    stop = time.time()

    print("Elapsed time: %f" % (stop - start))

    # to perform the plot of the background distribution
    random_aucs_per_size = random_aucs_per_signature.copy(deep=True)
    random_aucs_per_size.columns = [sign2size[s] for s in random_aucs_per_size.columns]
    random_aucs_per_size = random_aucs_per_size.loc[:, ~random_aucs_per_size.columns.duplicated()]
    sorted_cols = np.sort(random_aucs_per_size.columns.values)
    random_aucs_per_size[sorted_cols].to_csv(FINAL_RESULTS_PATH + "random_scores_UZ_Classification.csv")
