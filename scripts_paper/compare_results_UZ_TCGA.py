import sys
sys.path.append("../")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils.get_paths import FINAL_RESULTS_PATH
plt.close("all")

#mpl.rc('font',family='Roboto')

use_mln = True
use_both = False

results_tcga_mln = FINAL_RESULTS_PATH + "Results_MLN_centroid_TCGA_fixed_seed.csv"
results_tcga = FINAL_RESULTS_PATH + "results_TCGA_neg_versus_pos.csv"
results_uz = FINAL_RESULTS_PATH + "Classification_table_UZ_Ghent.csv"
result_seeding_testing = FINAL_RESULTS_PATH + "table1_paper.csv"

results_tcga = pd.read_csv(results_tcga, index_col=0)
results_uz = pd.read_csv(results_uz, index_col=0)
results_tcga_mln = pd.read_csv(results_tcga_mln, index_col=0)
result_seeding_testing = pd.read_csv(result_seeding_testing, index_col=0)

results_tcga.index = [s.replace("HALLMARK_", "").replace("_", " ").lower() for s in results_tcga.index]
results_uz.index = [s.replace("HALLMARK_", "").replace("_", " ").lower() for s in results_uz.index]
results_tcga_mln.index = [s.replace("HALLMARK_", "").replace("_", " ").lower() for s in results_tcga_mln.index]

short_name_dict = {"Activating invasion and metastasis": "invasion",
                   "Inducing angiogenesis": "angiogenesis",
                   "Deregulation of cellular energetics": "energetics",
                   "Enabling replicative immortality": "immortality",
                   "Cancer-immunology": "immunology",
                   'Genome instability': 'DNA instability',
                   'Sustaining proliferative signaling and evading growth suppressors': 'proliferation',
                   'Resisting cell death': 'cell death'
                   }

results_uz["Hallmark"] = [short_name_dict[hm] if hm in short_name_dict.keys() else hm
                          for hm in results_uz["Hallmark"].values]

sign2hm = dict(zip(results_uz.index.values, results_uz.Hallmark.values))

FDR_thresh_UZ = 1e-2
FDR_thresh_TCGA_mln = 1e-2
FDR_thresh_TCGA = 1e-5

sign_tcga = results_tcga.loc[results_tcga.FDR < FDR_thresh_TCGA].index.values
sign_uz = results_uz.loc[results_uz.FDR < FDR_thresh_UZ].index.values
sign_tcga_mln = results_tcga_mln.loc[results_tcga_mln.FDR < FDR_thresh_TCGA_mln].index.values

seeding_and_in_tcga = np.intersect1d(sign_tcga_mln, result_seeding_testing.index.values)

if use_both:
    sign_primary = np.union1d(sign_tcga, sign_tcga_mln)
    phase_1_signs = np.setdiff1d(sign_primary, sign_uz)
    phase_2_signs = np.intersect1d(sign_primary, sign_uz)
    phase_3_signs = np.setdiff1d(sign_uz, sign_primary)

    print("Signatures found for phase 1: %i" % len(phase_1_signs))
    print("Signatures found for phase 2: %i" % len(phase_2_signs))
    print("Signatures found for phase 3: %i" % len(phase_3_signs))


elif use_mln:
    phase_1_signs = np.setdiff1d(sign_tcga_mln, sign_uz)
    phase_2_signs = np.intersect1d(sign_tcga_mln, sign_uz)
    phase_3_signs = np.setdiff1d(sign_uz, sign_tcga_mln)

    print("Signatures found for phase 1: %i" % len(phase_1_signs))
    print("Signatures found for phase 2: %i" % len(phase_2_signs))
    print("Signatures found for phase 3: %i" % len(phase_3_signs))

else:
    phase_1_signs = np.setdiff1d(sign_tcga, sign_uz)
    phase_2_signs = np.intersect1d(sign_tcga, sign_uz)
    phase_3_signs = np.setdiff1d(sign_uz, sign_tcga)

    print("Signatures found for phase 1: %i" % len(phase_1_signs))
    print("Signatures found for phase 2: %i" % len(phase_2_signs))
    print("Signatures found for phase 3: %i" % len(phase_3_signs))

diff_mln_normal = np.setdiff1d(sign_tcga_mln, sign_tcga)
diff_normal_mln = np.setdiff1d(sign_tcga, sign_tcga_mln)

top_scoring_1 = results_tcga.loc[results_tcga.index.isin(phase_1_signs)]
top_scoring_2 = results_tcga.loc[results_tcga.index.isin(phase_2_signs)]
top_scoring_3 = results_uz.loc[results_uz.index.isin(phase_3_signs)]

hms1 = np.array([sign2hm[s] for s in phase_1_signs])
hms2 = np.array([sign2hm[s] for s in phase_2_signs])
hms3 = np.array([sign2hm[s] for s in phase_3_signs])

uniq_hms = np.union1d(hms1, np.union1d(hms2, hms3))

# create pie charts

hm2color_map = {'immunology': 'tab:orange',
                'invasion': 'k',
                'proliferation': 'g',
                'energetics': 'tab:purple',
                'angiogenesis': 'tab:red',
                'DNA instability': 'b',
                'cell death': 'tab:grey',
                'immortality': 'tab:blue'}

fig, axes = plt.subplots(ncols=3, figsize=(14, 10), sharey="row")
colors = [hm2color_map[h] for h in uniq_hms]

for i, sign in enumerate([hms1, hms2, hms3]):
    hallmarks, counts = np.unique(sign, return_counts=True)
    hm2count = dict(zip(hallmarks, counts))

    counts = np.asarray([hm2count[h] if h in hallmarks else 0 for h in uniq_hms])
    fractions = 100 * counts/counts.sum()

    axes[i].bar(np.arange(len(fractions)), fractions, color=colors, alpha=0.5, edgecolor=colors)

    axes[i].spines["left"].set_visible(False)
    axes[i].spines["right"].set_visible(False)
    axes[i].spines["bottom"].set_visible(False)
    axes[i].spines["top"].set_visible(False)

    axes[i].set_xticklabels([])
    axes[i].set_xticks([])


f1 = lambda c: plt.scatter([], [], marker="s",  edgecolor=c, facecolor=c, alpha=0.5)

handles = [f1(c) for c in colors]
labels = uniq_hms
fig.legend(handles, labels, loc=(0.05, -0.01), ncol=4, fontsize=19,
           handletextpad=0.1, frameon=False, columnspacing=3)

axes[0].set_ylabel("Percentage of identified signatures", fontsize=26)
#plt.savefig("results/figures_table_paper/histogram_figure1.svg", bbox_inches="tight", pad_inches=0)
plt.savefig(FINAL_RESULTS_PATH + "histogram_figure1.pdf", bbox_inches="tight", pad_inches=0)
plt.show()


