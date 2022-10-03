import sys
sys.path.append("../")
import pandas as pd
import numpy as np
from utils.CentroidClassifier import spearman_dist_mat
from utils.read_in_utils import get_expression, read_in_all_signatures, get_confirmed_patients
from scipy.stats import mannwhitneyu, wilcoxon
from utils.plots import sort_ids_by_patient_sample_type_number, get_patient2color
from identify_seeding_clone import get_inter_intra_dists
import matplotlib.pyplot as plt
from matplotlib.patches import BoxStyle
from matplotlib.path import Path
from scipy.stats import gaussian_kde
from utils.get_paths import FINAL_RESULTS_PATH

plt.close("all")


def get_relative_dists(exp, input_genes=None, source_tissue="RP", target_tissue="MLN"):
    identical_tissue = source_tissue == target_tissue

    exp = exp.copy(deep=True)

    if input_genes is not None:
        common_genes = np.intersect1d(exp.columns.values, input_genes)
        exp = exp[common_genes]

    rp_exp = exp.loc[[s for s in exp.index if source_tissue in s]]
    mln_exp = exp.loc[[s for s in exp.index if target_tissue in s]]

    dists = 1. - spearman_dist_mat(rp_exp.values, mln_exp.values)
    dists = pd.DataFrame(dists, index=rp_exp.index, columns=mln_exp.index)

    # centroid = mln_exp.mean(axis=0).values
    # centroid_dists = 1. - spearman_dist_mat(rp_exp.values, centroid)
    # centroid_dists = pd.Series(centroid_dists.flatten(), index=rp_exp.index)

    pat_mask = np.array([s.split("_")[1] for s in dists.index])[..., None] == \
               np.array([s.split("_")[1] for s in dists.columns])[None, ...]

    dists_temp = dists.copy(deep=True).values
    dists_temp[~pat_mask] = np.nan
    mean_intra_dists = np.nanmean(dists_temp, axis=1)
    mean_intra_dists = pd.Series(mean_intra_dists, index=dists.index.values)

    dists_temp = dists.copy(deep=True).values
    dists_temp[pat_mask] = np.nan
    mean_inter_dists = np.nanmean(dists_temp, axis=1)
    mean_inter_dists = pd.Series(mean_inter_dists, index=dists.index.values)

    relative_dists = mean_intra_dists/(mean_inter_dists + mean_intra_dists)

    return relative_dists


if __name__ == "__main__":
    confirmed_patients_new = get_confirmed_patients()

    shading_spacing = 0.4
    confirmed_patients_only = False

    exp = get_expression()
    exp = exp.transpose()

    signatures, index2biology = read_in_all_signatures(return_index2biology=True)

    relative_dists = {}

    signatures_of_interest = ["Focal adhesion", "Other genome instability"]
    # signatures_of_interest = ["Focal adhesion", "HALLMARK_HEDGEHOG_SIGNALING"]
    # signatures_of_interest = ["Focal adhesion", "EMT transcription factors"]

    for signature in signatures_of_interest:
        print(signature)
        common_genes = np.intersect1d(signatures.loc[signature].astype(str), exp.columns.values)

        if len(common_genes) > 5:
            relative_dists[signature] = get_relative_dists(exp, signatures.loc[signature].astype(str))

            intra_patient_df, inter_patient_df, pval = get_inter_intra_dists(exp, signatures.loc[signature].astype(str))
            print(pval)

            _, _, pval = get_inter_intra_dists(exp, signatures.loc[signature].astype(str),
                                               source_tissue="RP",
                                               target_tissue="RP")
            print(pval)

    plot_df = pd.DataFrame(relative_dists)

    if confirmed_patients_only:
        confirmed_patients = get_confirmed_patients()
        plot_df_mask = np.array([s.split("_")[1] in confirmed_patients for s in plot_df.index])
        plot_df = plot_df.loc[plot_df_mask]

    sort_idx = sort_ids_by_patient_sample_type_number(plot_df.index.values)
    plot_df = plot_df.loc[sort_idx]
    plot_df = plot_df.dropna()

    mannwhitneyu(plot_df[signatures_of_interest[0]].values,
                 plot_df[signatures_of_interest[1]].values,
                 alternative="greater")

    wilcoxon(plot_df[signatures_of_interest[0]].values,
                 plot_df[signatures_of_interest[1]].values)

    patient2color = get_patient2color()
    plot_df["Patient"] = [s.split("_")[1] for s in plot_df.index.values]

    patient_spacing = 2

    fig, axes = plt.subplots(ncols=2, figsize=(12, 5),  gridspec_kw={'width_ratios': [5, 1]},
                             sharey="row")

    plt.subplots_adjust(wspace=0.02)

    ax = axes[0]
    curr_pat, currx = "ID1", -1
    xposx = []
    shading_starts, shading_stops = [(-patient_spacing + shading_spacing) / 2.], []

    for sid, r in plot_df.iterrows():

        if curr_pat != r.Patient:
            currx += patient_spacing
            curr_pat = r.Patient
            shading_starts.append(currx - patient_spacing / 2. + shading_spacing / 2.)
            shading_stops.append(currx - patient_spacing / 2. - shading_spacing / 2.)

        else:
            currx += 1

        ax.scatter(currx, r[signatures_of_interest[0]],
                   marker="s", color="k",
                   edgecolor="k")

        ax.scatter(currx, r[signatures_of_interest[1]],
                   marker="s", color="none", edgecolor="k")

        xposx.append(currx)

    for pos in xposx:
        ax.axvline(x=pos, alpha=0.2, color="tab:gray")

    shading_stops.append(currx - 1 + patient_spacing)

    ax.set_xlabel("Patient ID", fontsize=16)
    ax.set_ylabel("Relative average distance", fontsize=16)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    pats = [s.split("_")[1] for s in plot_df.index]
    pat_counts = plot_df.groupby(pats, sort=False).count()["Patient"]
    patients = pat_counts.index.values
    xpos_labels, x_labels, gridlines_rows = [], [], []
    curr_pos = 0
    xlabel_positions = (np.asarray(shading_starts) + np.asarray(shading_stops))/2

    ax.set_xticks(xlabel_positions)
    ax.set_xticklabels(patients, rotation=0)
    ax.tick_params(axis=u'both', which=u'both', length=0)

    ax.set_xlim((shading_starts[0], shading_stops[-1]))
    ax.set_ylim((0, 1))

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["closer to\n same patient", "equal", "closer to \nother patients"])
    ax.axhline(y=0.5, linestyle="--", color="k", alpha=0.3)

    for i, (start, stop) in enumerate(zip(shading_starts, shading_stops)):
        pat = pat_counts.index.values[i]
        color = patient2color[pat]
        ax.axvspan(start, stop, color=color, alpha=0.2, ec=None)

    f1 = lambda c: plt.scatter([], [], marker="s", edgecolor="k", facecolor=c)

    handles = [f1(c) for c in ["k", "none"]]

    signature_names = [s.lower().replace("hallmark_", "").replace("_", " ").replace("other", "general")
                       for s in signatures_of_interest]
    labels = signature_names
    lgd = ax.legend(handles, labels, framealpha=1, loc=(0.22, 1.02), fontsize=16, ncol=2, columnspacing=3,
                    title="Signature", title_fontsize=14, handletextpad=0.1, frameon=False)

    plt.annotate(s='', xy=(0.129, .13), xytext=(0.129, .80), arrowprops=dict(arrowstyle='<->', mutation_scale=30),
                 xycoords='figure fraction')

    bins = np.linspace(0, 1, 1000)
    xs = (bins[1:] + bins[:-1])/2

    density_focal = gaussian_kde(plot_df[signatures_of_interest[0]].values)
    density_focal._compute_covariance()
    densities_focal = density_focal(xs)
    axes[1].plot(densities_focal, xs, linewidth=3, c="k")

    density_fao = gaussian_kde(plot_df[signatures_of_interest[1]].values)
    density_fao._compute_covariance()
    densities_fao = density_fao(xs)
    axes[1].plot(densities_fao, xs,  linewidth=0.5, c="k")

    xmin, xmax = -0.05, np.maximum(np.max(densities_fao), 1.03 * np.max(densities_focal))

    axes[1].set_xlim([xmin, xmax])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    axes[1].spines['bottom'].set_visible(False)
    axes[1].spines['left'].set_visible(False)

    axes[1].set_xticks([])
    axes[1].axhline(y=0.5, linestyle="--", color="k", alpha=0.3)
    axes[1].tick_params(axis=u'both', which=u'both', length=0)

    save_file = FINAL_RESULTS_PATH + "%s_%s.png" % \
                (signatures_of_interest[0].lower().replace(" ", ""),
                 signatures_of_interest[1].lower().replace(" ", ""))

    plt.savefig(save_file, dpi=1000, bbox_inches="tight")
    plt.show()

    plt.close()