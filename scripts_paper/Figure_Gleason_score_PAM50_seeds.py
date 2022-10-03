"""
Plots the Gleason scores per lesion and indicates which ones are selected to be seeding and which ones aren't.
"""

import sys
sys.path.append("../")
from utils.read_in_utils import get_clinical_info_UZ, get_varseeds
import pandas as pd
import numpy as np
from utils.read_in_utils import INPUT_PATH, get_confirmed_patients, FINAL_RESULTS_PATH, get_signature_voting_seeds
import matplotlib.pyplot as plt
from utils.plots import sort_ids_by_patient_sample_type_number, get_patient2color


plt.close("all")

clin_info = get_clinical_info_UZ()
shading_spacing = 0.4
pam50_scores = pd.read_csv(INPUT_PATH + "pam50_TPM.csv", index_col=0)

pam50_scores = pam50_scores.loc[["RP" in s.split("_")[-1] for s in pam50_scores.index]]
pam50_scores = pam50_scores[["Basal", "LumA", "LumB"]] / pam50_scores[["Basal", "LumA", "LumB"]].values.sum(axis=1,
                                                                                                            keepdims=True)
pam50_labels = pam50_scores.idxmax(axis=1).to_frame(name="pam50_label")
pamshort_2_long = {'Basal': 'Basal', 'LumA': 'Luminal A', 'LumB': 'Luminal B'}

patient2color = get_patient2color()

seeds = get_signature_voting_seeds()
var_seeds = get_varseeds()

confirmed_pats = get_confirmed_patients()

plot_df = clin_info["Gleason"].to_frame()
plot_df["Seed"] = plot_df.index.isin(seeds)
plot_df["Varseed"] = plot_df.index.isin(var_seeds)
plot_df = pd.merge(plot_df, pam50_labels, left_index=True, right_index=True)
sorted_ids = sort_ids_by_patient_sample_type_number(plot_df.index.values)
plot_df = plot_df.loc[sorted_ids]
plot_df["Gleason"] = plot_df["Gleason"].apply(lambda s: "+".join(s.split("+")[:2]))
uniq_gs = np.unique(plot_df.Gleason)
gs2num = {gs: i for i, gs in enumerate(uniq_gs)}
plot_df["GS_num"] = plot_df.Gleason.apply(lambda s: gs2num[s])

xs = np.arange(plot_df.shape[0])

pam50_markers = ["o", "s", "^"]
pam50_colors = ["r", "g", "b"]
pam50_labels = pd.unique(plot_df["pam50_label"])
pam50_marker_map = dict(zip(pam50_labels, pam50_markers))
pam50_color_map = dict(zip(pam50_labels, pam50_colors))

plot_df["markers"] = [pam50_marker_map[s] for s in plot_df.pam50_label]
plot_df["colors"] = [pam50_color_map[s] for s in plot_df.pam50_label]
plot_df["Patient"] = [s.split("_")[1] for s in plot_df.index]


patient_spacing = 2

fig, ax = plt.subplots(figsize=(12, 5))

curr_pat, currx = "ID1", -1
xposx = []
shading_starts, shading_stops = [(-patient_spacing + shading_spacing)/2.], []

for sid, r in plot_df.iterrows():

    if curr_pat != r.Patient:
        currx += patient_spacing
        curr_pat = r.Patient
        shading_starts.append(currx - patient_spacing/2. + shading_spacing/2.)
        shading_stops.append(currx - patient_spacing/2. - shading_spacing/2.)

    else:
        currx += 1

    xposx.append(currx)

shading_stops.append(currx - 1 + patient_spacing)

ax.set_xlabel("Patient ID", fontsize=16)
ax.set_ylabel("Gleason Score", fontsize=16)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(False)
ax.spines["left"].set_visible(False)

pats = [s.split("_")[1] for s in plot_df.index]
pat_counts = plot_df.groupby(pats, sort=False).count()["Gleason"]
xpos_labels, x_labels, gridlines_rows = [], [], []
curr_pos = 0
for pat, count in pat_counts.iteritems():
    xpos_labels.append(curr_pos + count/2)
    x_labels.append(str(pat))
    curr_pos += count + patient_spacing - 1

ax.set_xticks(xpos_labels)
ax.set_xticklabels(x_labels, rotation=45)

if confirmed_pats is not None:
    # confirmed_pats = [s.split("_")[1] for s in confirmed_seeds]
    labels = ax.get_xticklabels()
    for label in labels:
        if label.get_text() in confirmed_pats:
            label.set_fontweight('bold')

ax.set_yticks(np.arange(len(uniq_gs)))
ax.set_yticklabels(uniq_gs)
ax.set_xlim((shading_starts[0], shading_stops[-1]))

for i, (start, stop) in enumerate(zip(shading_starts, shading_stops)):
    pat = pat_counts.index.values[i]
    color = patient2color[pat]
    ax.axvspan(start, stop, color=color, alpha=0.2, ec=None)

plot_df["position"] = xposx
for sid, r in plot_df.iterrows():
    color = r.colors

    alpha = 0.6
    fc = "none"
    ec = "k"
    lw = 1

    if r.Seed:
        alpha = 1.
        fc = "k"

    if r.Varseed:
        ec = "tab:red"
        lw = 3

    ax.scatter(r.position, r.GS_num, marker=r.markers, color=color, edgecolor=ec, facecolor=fc, alpha=alpha, linewidth=lw)

f1 = lambda m, c: plt.scatter([], [], marker=m,  edgecolor="k", facecolor="none")

handles = [f1(m, c) for m, c in zip(pam50_markers, pam50_colors)]
labels = [pamshort_2_long[s] for s in pam50_labels]
lgd = ax.legend(handles, labels, framealpha=1, loc=(0.22, 1.02), fontsize=16, ncol=3, columnspacing=3,
                title="PAM50 subtype", title_fontsize=14, handletextpad=0.1, frameon=False)

plt.savefig(FINAL_RESULTS_PATH + "GS_pam50_seeding.pdf", dpi=1000, bbox_inches="tight")
plt.savefig(FINAL_RESULTS_PATH + "GS_pam50_seeding.png", dpi=1000, bbox_inches="tight")
plt.show()

#plt.close()