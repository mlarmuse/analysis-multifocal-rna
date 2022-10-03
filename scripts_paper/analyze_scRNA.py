import sys
sys.path.append("../")

import pandas as pd
import numpy as np
from utils.read_in_utils import read_in_all_signatures, get_expression
from utils.get_paths import INPUT_PATH, FINAL_RESULTS_PATH
import matplotlib.pyplot as plt
from utils.CentroidClassifier import CentroidClassifier, spearman_dist_mat
from utils.statistics import mwu
import seaborn as sns

# TODO: clean up the code here, all functions are redundant, they all should be imported


def plot_PCA(df, savename=None, xlabel="PC1", ylabel="PC2", centroids=None, labelmap=None, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = plt.gca()

    df = df.copy(deep=True)
    df.columns = [s.lower().replace("hallmark_", "").replace("_", "").replace("mln", "MLN") for s in df.columns]

    tissue_types = np.array([s.split("-")[-1] for s in df.index.values])

    tissue2int = {t: i for i, t in enumerate(np.unique(tissue_types))}

    markers = ['s', "o", 'x', "<", ">", "P", "^", "H", "+"]
    colors = ["tab:orange", "tab:blue", "tab:green",  "tab:red", "tab:purple", "tab:gray"]

    for i, arr in enumerate(df.values):
        marker = markers[tissue2int[tissue_types[i]]]
        color = colors[tissue2int[tissue_types[i]]]

        ax.scatter(arr[0], arr[1], c=color, marker=marker, s=100, edgecolors="k", alpha=0.7)

    f2 = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none", markersize=10, alpha=0.7)[0]

    handles = [f2(m, c) for m, c in zip(markers, colors)]

    labels = list(np.unique(tissue_types))

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xlabel(df.columns.values[0], fontsize=16)
    ax.set_ylabel(df.columns.values[1], fontsize=16)

    if centroids is not None:

        for c in range(centroids.shape[1]):
            print(centroids.iloc[:, c])
            ax.scatter(centroids.iloc[0, c], centroids.iloc[1, c],
                       marker=markers[c], color="k", s=500)

        handles += [f2(markers[c + len(labels)], "k") for c in range(centroids.shape[1])]
        labels += [c for c in centroids.columns]

    if labelmap is not None:
        labels = [labelmap[l] for l in labels]

    lgd = ax.legend(handles, labels, framealpha=1, bbox_to_anchor=(1.01, 0.97))

    if savename is not None:
        im_path = FINAL_RESULTS_PATH + savename
        print("Saving image to %s." % im_path)
        fig.savefig(im_path, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

    return ax


def predict_proba(centroid_df, X, dist_func, norm="own"):

    preds = dist_func(X, centroid_df)

    if norm.lower() == "pam50":
        preds[preds < 0.5] = 0

    return preds / preds.sum(axis=1, keepdims=True)


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


if __name__ == "__main__":

    # plot signatures
    imp_data = pd.read_hdf(INPUT_PATH + 'MAGIC_imputed_data.h5', 'df')
    epithelial_cells = pd.read_csv(INPUT_PATH + "epithelial_cells.csv",
                                    index_col=0)
    epithelial_cells = [s.replace(".", "-") for s in epithelial_cells.values.flatten()]
    imp_data = imp_data.loc[epithelial_cells]

    mln_pat_mask = [int(s.split("-")[1]) in [4, 6] for s in imp_data.index.values]
    mln_rp_data = imp_data.loc[mln_pat_mask]
    centroids_per_signature, exp, y, signatures = get_centroid_per_signature()

    signatures, index2biology = read_in_all_signatures(return_index2biology=True)
    results_uz = FINAL_RESULTS_PATH + "Classification_table_UZ_Ghent.csv"
    results_uz = pd.read_csv(results_uz, index_col=0)

    predictive_signatures = results_uz.index.values[results_uz.FDR < 0.01]

    soi = ["Fibroblasts", "HALLMARK_XENOBIOTIC_METABOLISM", "Focal adhesion", "Tumor Infiltration signature",
           "Growth factors invasion", "HALLMARK_SPERMATOGENESIS"]

    mln_mask = [s.split("-")[-1] == "4" for s in mln_rp_data.index.values]
    probs_signature = {}
    pvals = {}

    for signature in predictive_signatures:
        centroids = centroids_per_signature[signature]

        common_genes = np.intersect1d(centroids.index.values, mln_rp_data.columns.values)
        centroids = centroids.loc[common_genes]
        ccle_sign = mln_rp_data[common_genes]

        probs_signature[signature] = predict_proba(centroids.values.T, ccle_sign.values, spearman_dist_mat)[:, 0]
        U, pval = mwu(probs_signature[signature], mln_mask, alternative="less")
        pvals[signature] = pval

    sign_distances = pd.DataFrame(probs_signature, index=mln_rp_data.index)
    pvals_evolution_chen = pd.Series(pvals, index=predictive_signatures)


    plt.show()

    # plot the scatter plot with marginals
    plot_df = sign_distances.copy(deep=True)
    plot_df.columns = ["distance to %s MLN centroid" % s.lower() for s in sign_distances.columns]

    plot_df["Cell type"] = ["seeded" if s.split("-")[-1] == "4" else "pt" for s in plot_df.index.values]
    plot_df = plot_df.sort_values(by="Cell type")
    plt.figure()
    grid = sns.jointplot(data=plot_df, x="distance to hallmark_spermatogenesis MLN centroid",
                         y="distance to focal adhesion MLN centroid",
                         hue="Cell type", marginal_kws={"common_norm": False},
                         joint_kws={"alpha": 0.7,
                                    "edgecolors": "k"})

    grid.ax_joint.cla()

    ax = plot_PCA(plot_df[["distance to hallmark_spermatogenesis MLN centroid",
                                  "distance to focal adhesion MLN centroid"]],
                  labelmap={"4": "MLN cell", "6": "PT cell"},
                  ax=grid.ax_joint)

    plt.savefig(FINAL_RESULTS_PATH + "distance_scatter_mln_centroid_scrna.pdf", bbox_inches="tight")
    plt.show()

    plot_df = sign_distances
    plot_df["LNI status"] = np.array(mln_mask).astype(int)

    fig, axes = plt.subplots(nrows=len(soi), ncols=1, sharex="col")

    for i, signature in enumerate(soi):
        sname = signature.replace("_", " ").replace("HALLMARK ", "").lower()
        sns.kdeplot(data=plot_df, x=signature, hue="LNI status", ax=axes[i], common_norm=False)
        axes[i].spines["top"].set_visible(False)
        axes[i].spines["right"].set_visible(False)
        axes[i].spines["left"].set_visible(False)
        axes[i].spines["bottom"].set_color("tab:gray")

        axes[i].set_yticks([])
        axes[i].set_xticks([0.30, 0.4])
        axes[i].set_xticklabels(["further from MLN centroid", "closer to MLN centroid"])
        axes[i].xaxis.set_tick_params(width=.00001, color="w")

        #axes[i].xaxis.set_major_locator(mticker.NullLocator())
        #axes[i].yaxis.set_major_locator(mticker.NullLocator())
        axes[i].set_xlabel("")

        axes[i].set_ylabel(sname, rotation=0)
        axes[i].get_legend().remove()

    f2 = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none", markersize=10)[0]

    handles = [f2("s", v) for v in ["tab:blue", "tab:orange"]]
    plt.legend(handles, ["primary tumor", "lymph node"], loc=(0.14, 7.2), ncol=2, frameon=False)
    plt.savefig(FINAL_RESULTS_PATH + "distance_mln_centroid_scrna.pdf", bbox_inches="tight")
    plt.savefig(FINAL_RESULTS_PATH + "distance_mln_centroid_scrna.png", bbox_inches="tight")
    plt.savefig(FINAL_RESULTS_PATH + "distance_mln_centroid_scrna.svg", bbox_inches="tight")
    plt.show()