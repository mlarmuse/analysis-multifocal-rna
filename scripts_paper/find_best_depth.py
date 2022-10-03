import sys
sys.path.append("../")
import numpy as np
import pandas as pd
from utils.SeedsTargetsMatrix import SeedsTargetsMatrix
import matplotlib.pyplot as plt
from utils.get_paths import FINAL_RESULTS_PATH

min_supports = np.arange(1, 21, 1)
medians, min_fractions = [], []

for min_support in min_supports:
    pval_df = pd.read_csv(FINAL_RESULTS_PATH + "RP_2_MLN_pvals_depth%i.csv" % min_support,
                          index_col=0)

    stm = SeedsTargetsMatrix(pval_df)
    mins_per_samples = stm.get_min_values_per_target()

    medians.append(np.log10(mins_per_samples + 1e-300).median())

    min_ratio = stm.get_best_to_second_best_rank_score()
    # TODO: fix these bugs, min_ratio contains NaNs.
    min_fractions.append(min_ratio.mean())


medians = np.asarray(medians)

plt.figure()
plt.plot(min_supports, min_fractions, marker="s", c="k")
plt.xlabel("Minimal read support", fontsize=16)
plt.ylabel("Rank score ratio", fontsize=16)
ax = plt.gca()
ax.set_xticks([0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
plt.savefig(FINAL_RESULTS_PATH + "depth_gridsearch.png")
plt.show()

