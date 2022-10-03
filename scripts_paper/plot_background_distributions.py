"""
Uses the random scores and observed correct votes to identify the most important signatures and plots this.
"""

import sys
sys.path.append("../")

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from utils.get_paths import FINAL_RESULTS_PATH
from utils.read_in_utils import get_signature_sizes
from utils.misc import process_table
plt.close("all")


def plot_uncertainty(df, ax=None, y_label="AUC", bar_types="std", q=0.1):

    if ax is None:
        fig, ax = plt.subplots()

    col_ids = np.argsort(df.columns.values.astype(int))
    df = df.iloc[:, col_ids]

    means, sigma = df.mean(axis=0), df.std(axis=0)
    xs = df.columns.values.astype(int)

    # sort_ids = np.argsort(means.values)
    # means, sigma = means.values[sort_ids], sigma.values[sort_ids]

    ax.plot(xs, means, label="mean random score")

    if bar_types.lower() == "std":
        ax.fill_between(xs,
                        means - sigma,
                        means + sigma,
                        alpha=0.3,
                        label="Std")

    else:
        l_quant, u_quant = df.quantile(q=q), df.quantile(q=1 - q)
        ax.fill_between(xs,
                        l_quant,
                        u_quant,
                        alpha=0.3,
                        label=f"{q:.3f} quantile")

    ax.set_xlabel("Signature size", fontsize=18)
    ax.set_ylabel(y_label, fontsize=18)

    if y_label == "AUC":
        ax.axhline(0.5, linestyle="--")
        ax.annotate("random", xy=(400, 0.505), fontsize=14)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def get_pvalues(score_df, sign2size, randomscores):

    pvals = []
    for sign in score_df.index:
        print(sign)
        rscores = randomscores[str(sign2size[sign])].values
        score = score_df.loc[sign]

        pval = np.minimum((rscores >= score).sum(), (rscores <= score).sum()) / len(rscores)
        pvals.append(pval)

    return pvals


def get_quantiles(score_df, sign2size, randomscores, quant=0.01):
    l_quants = randomscores.quantile(q=quant)
    u_quants = randomscores.quantile(q=1. - quant)
    mus, sigmas = randomscores.mean(), randomscores.std()

    odict = {"Votes": [], "LQ": [], "UQ": [], "In_Interval": [], "Z-score": []}

    for sign in score_df.index:
        votes = score_df.loc[sign]
        lq, uq = l_quants.loc[str(sign2size[sign])], u_quants.loc[str(sign2size[sign])]
        odict["Votes"].append(votes)
        odict["LQ"].append(lq)
        odict["UQ"].append(uq)
        odict["In_Interval"].append((lq < votes) & (uq > votes))
        print(mus.loc[str(sign2size[sign])])
        print(sigmas.loc[str(sign2size[sign])])
        odict["Z-score"].append((float(votes) - mus.loc[str(sign2size[sign])]) / sigmas.loc[str(sign2size[sign])])

    return pd.DataFrame(odict, index=score_df.index)


if __name__ == "__main__":
    only_confirmed_patients = True

    MLN_df = pd.read_csv(FINAL_RESULTS_PATH + "random_scores_MLN_Classification.csv",
                         index_col=0)

    UZ_df = pd.read_csv(FINAL_RESULTS_PATH + "random_scores_UZ_Classification.csv",
                         index_col=0)

    fig, axes = plt.subplots(ncols=2)

    plot_uncertainty(MLN_df, axes[1], y_label="AUC MLN centroid")
    plot_uncertainty(UZ_df, axes[0], y_label="Worst class accuracy\n"
                                             "tissue classification")
    axes[1].set_ylim([0, 1])
    axes[0].set_ylim([0, 1])
    fig.tight_layout()

    axes[0].annotate("A.", xy=(-0.3, 1), xycoords='axes fraction', fontweight='extra bold')
    axes[1].annotate("B.", xy=(-0.3, 1), xycoords='axes fraction', fontweight='extra bold')
    #plt.savefig("results/figures_table_paper/background_distribution.png", dpi=600)
    plt.show()

    sign2size, uniq_sizes = get_signature_sizes()

    tag = ""
    if only_confirmed_patients:
        tag = "_confirmed_patients"

    random_scores = pd.read_csv(FINAL_RESULTS_PATH + "randomscores_seeding_signature_size_all_genes%s.csv" % tag, index_col=0)
    observed_votes = pd.read_csv(FINAL_RESULTS_PATH + "scores_seeding_signatures%s.csv" % tag, index_col=0)
    real_counts = observed_votes["Vote score"]

    random_scores_indexlesions = pd.read_csv(FINAL_RESULTS_PATH + "randomscores_index_lesion_signature_size_all_genes%s.csv"%tag,
                                             index_col=0)
    random_scores_indexlesions.mean()

    observed_votes["P_value"] = get_pvalues(real_counts, sign2size, random_scores)

    quantile_df = get_quantiles(real_counts, sign2size, random_scores)
    quantile_df.sort_values(by="Z-score", ascending=False)

    output_table = pd.merge(quantile_df, observed_votes, left_index=True, right_index=True)
    output_table["signature size"] = [sign2size[s] for s in output_table.index]
    output_table = output_table.sort_values(by="Z-score", ascending=False)

    output_table = output_table[["Votes", "q-value", "Z-score", "signature size"]]
    output = output_table.astype({"Votes": int, "q-value": float, "Z-score": float, "signature size": int})
    output = output.sort_values(by="Z-score", ascending=False)
    output[["q-value", "Z-score"]] = process_table(output[["q-value", "Z-score"]])
    output.index = [s.replace("HALLMARK_", "").replace("_", " ").lower() for s in output.index]
    output.to_csv(FINAL_RESULTS_PATH + "table_top_signaures_seeding_test%s.csv"%tag)
    table1 = output.loc[(output["q-value"].astype(float) < 0.01) |(output["q-value"].astype(float) > 0.99)].head(10)
    table1.to_csv(FINAL_RESULTS_PATH + "table1_paper.csv")
    # TODO: change script using this quantile_df

    quantile_df.to_csv(FINAL_RESULTS_PATH + "quantiles_seeds_signature_size.csv", index=True)

    included_signatures = table1.index.values

    fig, ax = plt.subplots(figsize=(15, 10))
    ann_fs = 15
    quant = 0.005
    random_scores = random_scores.loc[:, [int(s) > 5 for s in random_scores.columns]]
    random_scores = random_scores.loc[:, [int(s) < 205 for s in random_scores.columns]]

    plot_uncertainty(random_scores, ax=ax, y_label="Correct votes", bar_types="perc", q=quant)

    ax.set_xlim([0, 205])

    l_quants = random_scores.quantile(q=quant)
    u_quants = random_scores.quantile(q=1.-quant)

    plotted_signatures = []

    for sign in output_table.index.values:

        ovotes = real_counts.loc[sign]
        sign_size = sign2size[sign]
        zscore = output_table.loc[sign]["Z-score"]
        qval = output_table.loc[sign]["q-value"]

        if sign_size < 205:
            uq, lq = u_quants.loc[str(sign_size)], l_quants.loc[str(sign_size)]

        else:
            uq, lq = 17, 0

        valid_qval = (qval < 0.01) | (qval > 0.99)

        if (np.abs(zscore) > 2.3) and valid_qval:
            print(sign)
            print((sign_size, ovotes))
            plotted_signatures.append(sign)
            stext = sign.replace("HALLMARK_", "").replace("_", " ").lower()
            stext = stext.replace("cluster1 ", "").replace("genes", "")

            if stext == "548 genes signature":
                ax.annotate(stext, (sign_size - 30, ovotes+0.25), fontsize=ann_fs)

            elif stext == "hypoxia":
                ax.annotate(stext, (sign_size - 11, ovotes + 0.2), fontsize=ann_fs)

            elif stext == "monocytic lineage":
                ax.annotate(stext, (sign_size - 1, ovotes - 0.4), fontsize=ann_fs)

            elif stext == "b-lineage":
                ax.annotate(stext, (sign_size - 5, ovotes+0.25), fontsize=ann_fs)

            elif stext == "antigenpresentationtcellactivation":
                ax.annotate(stext, (sign_size - 5, ovotes-0.4), fontsize=ann_fs)

            elif stext == "cdc42 signaling":
                ax.annotate(stext, (sign_size - 4, ovotes+0.25), fontsize=ann_fs)

            elif stext == "il13":
                ax.annotate(stext, (sign_size - 4, ovotes+0.8), fontsize=ann_fs)

            elif stext == "pi3k akt mtor signaling":
                ax.annotate(stext, (sign_size - 35, ovotes+0.25), fontsize=ann_fs)

            elif stext == "mismatch repair":
                ax.annotate(stext, (sign_size - 4, ovotes-0.5), fontsize=ann_fs)

            elif stext == "growth transcriptional factors":
                ax.annotate(stext, (sign_size, ovotes - 0.4), fontsize=ann_fs)

            elif stext == "hedgehog signaling":
                ax.annotate(stext, (sign_size - 25, ovotes + 0.2), fontsize=ann_fs)

            elif stext == "tnf receptor pathway":
                ax.annotate(stext, xy=(sign_size, ovotes), xytext=(sign_size, ovotes - 2.5), fontsize=ann_fs,
                            arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "emt transcription factors":
                ax.annotate(stext, xy=(sign_size, ovotes), xytext=(sign_size, ovotes + 0.2), fontsize=ann_fs,
                            arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "glycogen metabolism":
                ax.annotate(stext, xy=(sign_size, ovotes), xytext=(sign_size, ovotes - 3), fontsize=ann_fs,
                            arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "dna repair":
                ax.annotate(stext, xy=(sign_size, ovotes), xytext=(sign_size, ovotes - 2.5), fontsize=ann_fs,
                            arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "growth factors and receptors":
                ax.annotate(stext, xy=(sign_size, ovotes), xytext=(sign_size, ovotes - 2), fontsize=ann_fs,
                            arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "focal adhesion":
                ax.annotate(stext, xy=(sign_size, ovotes),
                            xytext=(sign_size, ovotes - 2), fontsize=ann_fs,
                            arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "growth factors immortality":
                ax.annotate(stext, xy=(sign_size, ovotes), xytext=(sign_size, ovotes - 1), fontsize=ann_fs,
                            arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "rho signaling":
                ax.annotate(stext, xy=(sign_size, ovotes), xytext=(sign_size + 1, ovotes - 0.5), fontsize=ann_fs)

            elif stext == "nepc signature":
                ax.annotate(stext, (sign_size, ovotes - 0.5), fontsize=ann_fs)

            elif stext == "apical surface":
                ax.annotate(stext, (sign_size, ovotes - 1), fontsize=ann_fs)

            elif stext == "oxidative phosphorylation":
                ax.annotate(stext, (sign_size - 20, ovotes - 0.5), fontsize=ann_fs)

            elif stext == "xenobiotic metabolism":
                ax.annotate(stext, (sign_size - 20, ovotes - 1), fontsize=ann_fs)

            elif stext == "regulation":
                ax.annotate(stext, (sign_size, ovotes - 0.3), fontsize=ann_fs)

            elif stext == "pancreas beta cells":
                ax.annotate(stext, (sign_size - 10, ovotes - 0.3), fontsize=ann_fs)

            elif stext == "receptors":
                ax.annotate(stext, (sign_size + 1, ovotes - 0.3), fontsize=ann_fs)

            elif stext == "tumor infiltration signature":
                ax.annotate(stext, xy=(sign_size, ovotes),
                            xytext=(sign_size, ovotes - 1.5),
                            fontsize=ann_fs, arrowprops={'width': 0.5, 'headwidth': 0})

            elif stext == "other genome instability":
                ax.annotate(stext.replace("other", "general"), (sign_size, ovotes + 0.25), fontsize=ann_fs)
            else:
                ax.annotate(stext, (sign_size, ovotes + 0.25), fontsize=ann_fs)

            ax.scatter(sign_size, ovotes, alpha=0.5, c="tab:orange")

        elif sign in included_signatures:
            stext = sign.replace("HALLMARK_", "").replace("_", " ").lower()
            stext = stext.replace("cluster1 ", "")
            ax.scatter(sign_size, ovotes, alpha=0.5, c="tab:grey")
            if stext == "hypoxia":
                ax.annotate(stext, (sign_size - 11, ovotes + 0.3), fontsize=ann_fs, color="tab:grey")

            elif stext == "oxidative phosphorylation":

                ax.annotate(stext, (sign_size - 20, ovotes - 0.5), fontsize=ann_fs,  color="tab:grey")

            elif stext == "xenobiotic metabolism":
                ax.annotate(stext, (sign_size - 20, ovotes - 1), fontsize=ann_fs,  color="tab:grey")

            else:
                ax.annotate(stext, (sign_size, ovotes + 0.25), fontsize=ann_fs, color="tab:grey")

    include_index_lesions = True

    if include_index_lesions:
        mean_rscores_il = random_scores_indexlesions.mean(axis=0)
        mean_rscores_il.index = mean_rscores_il.index.values.astype(int)
        mean_rscores_il = mean_rscores_il.loc[(mean_rscores_il.index.values > 5) & (mean_rscores_il.index.values < 205)]
        mean_rscores_il = mean_rscores_il.sort_index()

        ax.plot(mean_rscores_il.index, mean_rscores_il.values, label="mean score index lesions")

    ax.legend(fontsize=16, loc="lower right")

    plt.savefig(FINAL_RESULTS_PATH + "quantiles_signature_sizes_seeds_all_genes.png", dpi=1000, bbox_inches="tight")
    plt.show()

    plt.close("all")
