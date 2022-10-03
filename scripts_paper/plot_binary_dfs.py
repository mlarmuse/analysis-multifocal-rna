import sys
sys.path.append("../")
import pandas as pd
from utils.read_in_utils import read_in_all_signatures
from utils.get_paths import FINAL_RESULTS_PATH
import numpy as np
import matplotlib as mpl
from matplotlib import cm
from utils.plots import hm2color, sort_sample_ids
import matplotlib.pyplot as plt
import seaborn as sns
plt.close("all")

show_hm = True
zscores = pd.read_csv(FINAL_RESULTS_PATH + "Results_MLN_centroid_TCGA_fixed_seed.csv", index_col=0)
predictive_signatures = zscores.loc[zscores["FDR"] < 0.01].index

signatures, index2biology = read_in_all_signatures(return_index2biology=True)

binary_dfs = pd.read_csv(FINAL_RESULTS_PATH  + "binary_df.csv", index_col=0)
sorted_sample_ids = sort_sample_ids(binary_dfs.index.values)
binary_dfs = binary_dfs.loc[sorted_sample_ids]

plot_df = binary_dfs[predictive_signatures]
hm2color_map = hm2color()
hallmarks = np.array([index2biology[hm] for hm in plot_df.columns])
plot_df = plot_df.iloc[:, np.argsort(hallmarks)]

patients = [s.split("_")[1] for s in plot_df.index.values]
uniq_patients = pd.unique(patients)

col_colors = pd.Series({hm: hm2color_map[index2biology[hm]] for hm in plot_df.columns})

viridis = cm.get_cmap('tab20', 17)

viridis_list = viridis(range(17))
patients = [s.split("_")[1] for s in plot_df.index.values]
patient2color = {p: viridis_list[i] for i, p in enumerate(pd.unique(patients))}
row_colors = pd.Series({pat: patient2color[pat.split("_")[1]] for pat in plot_df.index.values})

uniq_colors_2_int = {c: i for i, c in enumerate(pd.unique(col_colors))}
plot_df = plot_df * np.array([uniq_colors_2_int[g] + 1 for g in col_colors])

cmap_list = ['w'] + [c for i, c in enumerate(pd.unique(col_colors))]
cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmap_list, len(cmap_list))

gr = sns.clustermap(plot_df,
                    col_cluster=False,
                    row_cluster=False,
                    xticklabels=False,
                    yticklabels=False,
                    annot_kws={"fontsize": 'x-small'},
                    col_colors=col_colors,
                    row_colors=row_colors,
                    cmap=cmap,
                    linewidths=0.5,
                    linecolor='w',
                    cbar_pos=(0.93, .1, .03, .6),
                    )

pat_numbers = np.array([int(''.join(c for c in s.split("_")[1] if c.isdigit())) for s in row_colors.index])

row_counts = row_colors.groupby(pat_numbers).count()
xpos_labels, x_labels, gridlines_rows = [], [], []
curr_pos = 0

for pat, count in row_counts.iteritems():
    xpos_labels.append(curr_pos + count/2)
    x_labels.append("ID" + str(pat))
    curr_pos += count

if show_hm:
    hallmarks = [index2biology[s] for s in col_colors.index]
    col_counts = col_colors.groupby(hallmarks).count()

    ypos_labels, y_labels = [], []
    curr_pos = 0
    for hm, count in col_counts.iteritems():
        ypos_labels.append(curr_pos + count/2)
        y_labels.append(hm)
        curr_pos += count

gr.cax.set_visible(False)
gr.ax_col_dendrogram.set_visible(False)
gr.ax_row_dendrogram.set_visible(False)
gr.cax.set_axis_off()
gr.ax_heatmap.grid(which='major', axis='both', linestyle='-', linewidth='0.5', color='w')

short_name_dict = {"Activating invasion and metastasis": "invasion",
                   "Inducing angiogenesis": "angiogenesis",
                   "Deregulation of cellular energetics": "energetics",
                   "Enabling replicative immortality": "immortality",
                   "Cancer-immunology": "immunology",
                   'Genome instability': 'DNA instability',
                   'Sustaining proliferative signaling and evading growth suppressors': 'proliferation',
                   'Resisting cell death': 'cell death'
                   }

gr.ax_row_colors.set_yticks(xpos_labels)
gr.ax_row_colors.set_yticklabels(x_labels)
gr.ax_row_colors.set_xticks([])

if show_hm:
    ax2 = gr.ax_col_colors.twiny()
    ax2.set_xticks(np.array(ypos_labels)/col_colors.shape[0])
    ax2.set_xticklabels([short_name_dict[s] for s in y_labels], rotation=45, ha="left")

else:
    ax2 = gr.ax_col_colors.twiny()
    ypos_labels = np.arange(col_colors.shape[0]) + 0.5
    ax2.set_xticks(ypos_labels/col_colors.shape[0])

    ylabels = [s.replace("_", " ").replace("HALLMARK", "").lower() for s in col_colors.index.values]
    ax2.set_xticklabels(ylabels, rotation=30, ha="left")

gr.ax_col_colors.set_xticks([])
gr.ax_col_colors.set_yticks([])

gr.ax_heatmap.set_yticks(np.arange(gr.ax_heatmap.get_ylim()[0]))
a = gr.ax_heatmap.get_ygridlines()
counter = 0

grid_pos = np.cumsum(row_counts.values)

for i in range(len(a)):
    if i in grid_pos:
        a[i].set_color('grey')
        a[i].set_linewidth(0.8)

    else:
        a[i].set_color('w')
        a[i].set_linewidth(0.5)


ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.spines["bottom"].set_visible(False)

for tic in ax2.xaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False

for tic in gr.ax_row_colors.yaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False
    tic.get_loc()

for tic in gr.ax_heatmap.yaxis.get_major_ticks():
    tic.tick1On = tic.tick2On = False


gr.ax_row_colors.set_yticklabels(x_labels)

fig = gr.fig

pos1 = gr.ax_heatmap.get_position(original=True)
pos_rc = gr.ax_row_colors.get_position()
pos_new_ax = [pos1.x0 + pos1.width * 1.05, pos1.intervaly[0],  0.2, pos1.intervaly[-1] - pos1.intervaly[0]]
barplot_ax = fig.add_axes(pos_new_ax)

sums_per_pat = (plot_df > 0).sum(axis=1)
barwidth = 0.8 * 1/plot_df.shape[0]
ys = np.arange(plot_df.shape[0])/plot_df.shape[0]
barplot_ax.barh(ys[::-1] + barwidth/2, sums_per_pat.values, color=row_colors.to_list(), height=barwidth)
barplot_ax.set_ylim([0, 1])
barplot_ax.set_xticks([])
barplot_ax.set_yticks([])


barplot_ax.spines["left"].set_visible(False)
barplot_ax.spines["top"].set_visible(False)
barplot_ax.spines["right"].set_visible(False)
barplot_ax.spines["bottom"].set_visible(False)


plt.savefig(FINAL_RESULTS_PATH + "binary_dfs_seeding.pdf", bbox_inches="tight", pad_inches=0)
plt.show()

plt.close()