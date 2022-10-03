"""
utilities to plot
"""

import pandas as pd
import seaborn as sns
from scipy.spatial.distance import pdist

import scipy.cluster.hierarchy as hc
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.patches as patches
import numpy as np
from matplotlib import cm

from utils.get_paths import RESULTS_PATH, FINAL_RESULTS_PATH
from utils.misc import sort_ids_by_patient_sample_type_number
from scipy.stats import spearmanr
from matplotlib.patches import Patch

import copy
plt.close("all")


def plot_hist(sample_id, dist):

    patient_id = sample_id.split("_")[1]
    v = dist[sample_id].values

    mask = np.array([s != sample_id for s in dist.index.values])
    v = v[mask]
    mask = np.array([s.split("_")[1] == patient_id
                     for s in dist.index.values[mask]])

    bins = np.linspace(v.min(), v.max(), num=20)

    plt.figure()
    ax = plt.gca()
    ax.hist(v[mask], bins=bins, label="same patient", alpha=0.5)
    ax.hist(v[~mask], bins=bins, label="different patient", alpha=0.5)
    plt.legend()
    plt.show()


def plot_background(rp2ln):
    same_pat_mask = np.array([s.split("_")[1] for s in rp2ln.index.values])[..., None] == \
                    np.array([s.split("_")[1] for s in rp2ln.columns.values])[None, ...]

    background = rp2ln.values[~same_pat_mask]
    real = rp2ln.values[same_pat_mask]

    bins = np.linspace(0, 1, 50)

    plt.figure()
    ax = plt.gca()
    ax.hist(background, bins=bins, label="Background", alpha=0.5)
    ax.hist(real, bins=bins, label="Truth", alpha=0.5)
    plt.legend()
    plt.show()


def plot_clustermap(signature_values, index2biology=None, cluster_rows=True, savename="heatmap_data",
                    col_colors=None, cclust_data=None, show_legend=True):

    plot_biologies = True
    if index2biology is None:
        index2biology = {s: "not known" for s in signature_values.index.values}
        plot_biologies = False

    if col_colors is None: # should be a pd DF otherwise
        col_colors, cclust_data = get_col_color_df(signature_values)

    elif not isinstance(col_colors, pd.DataFrame):
        raise IOError("col_colors should be a pd DF.")

    if cclust_data is None:
        cclust_data = col_colors.copy()

    dist = pdist(signature_values.transpose().values, metric="euclidean")
    #dist = squareform(1. - signature_values.corr().values)

    numclust = 3
    linkage_mat = hc.linkage(dist, method='ward')
    clusters = hc.fcluster(linkage_mat, numclust, criterion='maxclust')

    tab_cm = cm.get_cmap('tab20', numclust)
    tab_cm_list = tab_cm(range(numclust))
    cluster2color = {c: tab_cm_list[i] for i, c in enumerate(np.unique(clusters))}
    cluster_colors = [cluster2color[c] for c in clusters]
    col_colors["Cluster"] = cluster_colors
    cclust_data["Cluster"] = clusters

    biologies = [index2biology[index] for index in signature_values.index]
    uniq_biologies = set(biologies)
    lut = dict(zip(set(uniq_biologies), sns.hls_palette(len(set(uniq_biologies)), l=0.5, s=0.8)))

    rcluster_data = 0

    if cluster_rows:
        dist = pdist(signature_values.values, metric="euclidean")

        numclust = 6
        rlinkage_mat = hc.linkage(dist, method='ward')
        rclusters = hc.fcluster(rlinkage_mat, numclust, criterion='maxclust')

        tab_cm = cm.get_cmap('tab20', numclust)
        tab_cm_list = tab_cm(range(numclust))
        cluster2color = {c: tab_cm_list[i] for i, c in enumerate(np.unique(rclusters))}
        rcluster_colors = [cluster2color[c] for c in rclusters]
        biology_colors = [lut[b] for b in biologies]

        if plot_biologies:
            row_colors = pd.DataFrame({"Hallmark": biology_colors, "Cluster": rcluster_colors},
                                      index=signature_values.index)

        else:
            row_colors = pd.DataFrame({"Cluster": rcluster_colors},
                                      index=signature_values.index)

        rcluster_data = pd.DataFrame({"Hallmark": biologies, "Cluster": rclusters, "Color": biology_colors},
                                     index=signature_values.index)

    else:
        row_colors = pd.Series(biologies,
                               index=signature_values.index).map(lut)

        sort_ids = np.argsort(biologies)
        signature_values = signature_values.iloc[sort_ids, :]
        rlinkage_mat = None

        # to fix the ylabels:

        v, biology_counts = np.unique(biologies, return_counts=True)
        pos = 0
        ypos, bio_labels = [], []

        for counts, v in zip(biology_counts, v):
            ypos.append(pos + 1. * counts / 2)
            pos += counts
            bio_labels.append(v + "     ")

    # Create the colormap using the dictionary
    GnRd = get_GnRd_cm()

    fig = plt.figure()
    gr = sns.clustermap(signature_values,
                        row_cluster=cluster_rows,
                        col_cluster=True,
                        col_linkage=linkage_mat,
                        row_linkage=rlinkage_mat,
                        cmap=GnRd,
                        xticklabels=False,
                        yticklabels=False,
                        annot_kws={"fontsize": 'x-small'},
                        col_colors=col_colors,
                        row_colors=row_colors,
                        cbar_pos=(0.93, .1, .03, .6),
                        #cbar_kws=dict(use_gridspec=False, loc="right"),
                        method='ward')

    fontsize = 12
    fontweight = 'bold'
    fontproperties = {'family': 'sans-serif', 'weight': fontweight, 'size': fontsize}

    if show_legend:
        print("adding legends")
        if plot_biologies:
            handles = [Patch(facecolor=lut[name]) for name in lut]
            gr.fig.legend(handles, lut, title='Hallmarks', loc=(0.04, .75)) #bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure,

        color2tissue = {"LN": "tab:blue", "RP": "tab:orange", "AN": "tab:green"}
        handles = [Patch(facecolor=color2tissue[name]) for name in color2tissue]
        gr.fig.legend(handles, color2tissue, title='Tissue', loc=(0.87, .87)) #bbox_to_anchor=(1, 1), bbox_transform=plt.gcf().transFigure,

    if not cluster_rows:
        ax = gr.ax_heatmap
        ax.set_yticks(ypos)
        ax.set_yticklabels(bio_labels, fontdict=fontproperties)
        ax.tick_params(right=False, top=False, left=False, labelright=False, labeltop=False, labelleft=True)
        gr.ax_row_colors.set_xticks([])

        if savename is not None:
            plt.savefig(RESULTS_PATH + "/figures/" + savename + ".png", bbox_inches="tight")

    else:
        if savename is not None:
            plt.savefig(RESULTS_PATH + "/figures/" + savename + "_row_clustered.png", bbox_inches="tight")

    return rcluster_data, cclust_data


def get_GnRd_cm():
    # This dictionary defines the colormap
    cdict = {'red': ((0.0, 0.0, 0.0),  # no red at 0
                     (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                     (1.0, 0.8, 0.8)),  # set to 0.8 so its not too bright at 1

             'green': ((0.0, 0.8, 0.8),  # set to 0.8 so its not too bright at 0
                       (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                       (1.0, 0.0, 0.0)),  # no green at 1

             'blue': ((0.0, 0.0, 0.0),  # no blue at 0
                      (0.5, 1.0, 1.0),  # all channels set to 1.0 at 0.5 to create white
                      (1.0, 0.0, 0.0))  # no blue at 1
             }

    return colors.LinearSegmentedColormap('GnRd', cdict)


def get_col_color_df(signature_values, cm_used="tab20"):
    patients = np.array([s.split("_")[1] for s in signature_values.columns.values])
    patient2color = get_patient2color()
    patient_colors = pd.Series([patient2color[p] for p in patients], index=signature_values.columns)

    tissue_colors = pd.Series(["tab:blue" if "LN" in p else "tab:orange" if "RP" in p else "tab:green"
                               for p in signature_values.columns.values], index=signature_values.columns)

    return pd.DataFrame({"Patient": patient_colors,
                         "Tissue": tissue_colors}), \
           pd.DataFrame({"Patient": signature_values.columns,
                         "tissue": ["LN" if "LN" in p else "NL" if "NL" in p else "RP"
                               for p in signature_values.columns.values]})


def plot_scatter(df, tag=None, tissue_types=None, savename=None, colorpats=True, centroids=None, plot_arrows=False):
    '''

    :param df:
    :param tag:
    :param tissue_types:
    :param savename:
    :param colorpats:
    :param centroids: A dict containing the centroid coordinates in 2D as values and the tissue type as key
    :return:
    '''
    fig = plt.figure()
    ax = plt.gca()

    viridis = cm.get_cmap('tab20', 17)

    viridis_list = viridis(range(17))

    if tissue_types is None:
        tissue_types = np.array(['PT' if 'RP' in s.split("_")[-1] else 'MLN' if 'LN' in s.split("_")[-1]
                                 else 'MET' if 'MET' in s.split("_")[-1] else 'AN'
                                 for s in df.index.values])

    patients = np.array([s.split("_")[1] for s in df.index.values])
    uniq_pats = pd.unique(patients)
    tissue2int = {t: i for i, t in enumerate(np.unique(tissue_types))}
    if colorpats:
        pat2int = {t: i for i, t in enumerate(uniq_pats)}

    else:
        pat2int = {t: 0 for i, t in enumerate(uniq_pats)}

    markers = ["o", 's', 'x', "<", ">", "P", "^", "H", "+"]

    for i, arr in enumerate(df.values):
        marker = markers[tissue2int[tissue_types[i]]]
        color = viridis_list[pat2int[patients[i]]][None, ...]

        ax.scatter(arr[0], arr[1], c=color, marker=marker)

    if centroids is not None:
        for tissue, centroid in centroids.items():
            marker = markers[tissue2int[tissue]]
            ax.scatter(centroid[0], centroid[1], facecolors="k", edgecolors='k', marker=marker, s=100)

        if plot_arrows:
            style = "Simple, tail_width=0.5, head_width=3, head_length=5"
            kw = dict(arrowstyle=style, color="k", shrinkA=8, shrinkB=8)

            a1 = patches.FancyArrowPatch(centroids["AN"], centroids["PT"],
                                         connectionstyle="arc3,rad=.5",
                                         mutation_scale=2, **kw)
            plt.gca().add_patch(a1)
            a2 = patches.FancyArrowPatch(centroids["PT"], centroids["MLN"],
                                         connectionstyle="arc3,rad=.2",
                                         mutation_scale=2, **kw)
            plt.gca().add_patch(a2)

    f1 = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    f2 = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none", markersize=15)[0]

    handles = [f2("s", v) for v in viridis_list]
    handles += [f1(m, "k") for m in markers]

    labels = np.hstack((uniq_pats, np.unique(tissue_types)))

    if tag is not None:
        ax.set_xlabel(tag + " 1")
        ax.set_ylabel(tag + " 2")

    else:
        tag = str(df.columns.values[0]) + "_" + str(df.columns.values[1])
        ax.set_xlabel(df.columns.values[0])
        ax.set_ylabel(df.columns.values[1])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    lgd = ax.legend(handles, labels, framealpha=1, bbox_to_anchor=(1.01, 1.06))

    if savename is not None:
        im_path = FINAL_RESULTS_PATH + savename
        print("Saving image to %s." % im_path)
        fig.savefig(im_path, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=600)

    plt.show()


def plot_scores_per_patient(score_series, ax=None, tissue_types=None, sort_patients_axis=True,
                            random_offset_per_sample=True, offset_interval=0.2,
                            special_group=None,
                            ylabel=None,
                            random_seed=42,
                            save_path=None):
    """
    Plot heterogeneity according to Kumar paper.
    :param score_series:
    :param ax:
    :param tissue_types:
    :param sort_patients_axis:
    :param random_offset_per_sample:
    :param offset_interval:
    :param ylabel:
    :param random_seed:
    :param save_path:
    :return:
    """

    if special_group is None:
        special_group = []

    np.random.seed(random_seed)
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if ylabel is None:
        ylabel = score_series.name

    viridis = cm.get_cmap('tab20', 17)
    marker_types = np.array(["o", 's', 'x', "<", ">", "P", "^", "H", "+"])
    viridis_list = viridis(range(17))

    if tissue_types is None:
        tissue_types = np.array(['PT' if 'RP' in s.split("_")[-1] else 'LN' if 'LN' in s.split("_")[-1] else 'AN'
                                 for s in score_series.index.values])

    patients = np.array([s.split("_")[1] for s in score_series.index.values])

    if sort_patients_axis:
        uniq_patients = sort_ids_by_patient_sample_type_number(score_series.index.values)

    else:
        uniq_patients = np.unique(patients)

    tissue2int = {t: i for i, t in enumerate(np.unique(tissue_types))}
    pat2int = {t: i for i, t in enumerate(uniq_patients)}

    colors = [viridis_list[pat2int[pat]] for pat in patients]
    markers = [marker_types[tissue2int[t]] for t in tissue_types]

    xs = [pat2int[pat] for pat in patients]
    offset_per_sample = 0

    for i, v in enumerate(score_series.values.flatten()):
        sample = score_series.index.values[i]
        if random_offset_per_sample:
            random_offset_per_sample = (np.random.random() - .5) * offset_interval / .5

        tissue = 'PT' if 'RP' in sample.split("_")[-1] else 'LN' if 'LN' in sample.split("_")[-1] else 'AN'
        tissue_int = tissue2int[tissue]
        marker = markers[tissue_int]

        if sample in special_group:
            ec = "k"
            marker = "s"

        else:
            ec = None

        ax.scatter(xs[i] + random_offset_per_sample, v, c=colors[i], marker=marker, edgecolors=ec)

    ax.set_xlabel("Patient ID", fontsize=16)
    ax.set_ylabel(ylabel=ylabel, fontsize=16)

    ax.set_xticks(np.arange(len(uniq_patients)))
    ax.set_xticklabels(uniq_patients)

    f1 = lambda m, c: plt.plot([], [], marker=m, c=c, ls="none")[0]

    handles = [f1(m, "k") for m in marker_types]
    labels = ["PT", "Index lesion"]
    lgd = ax.legend(handles, labels, framealpha=1)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def plot_activities_2D(plot_df, ax=None,
                       xlabel="Signature 1 activity",
                       ylabel="Signature 2 activity",
                       save_path=None):

    labels = [s.split("_")[-1][3:] if "MLN" in s else s.split("_")[-1][2:] for s in plot_df.index.values]
    colors = np.array(["tab:green" if "NL" in s else "tab:orange" if "LN" in s else "tab:blue" for s in plot_df.index])
    #legend_text = np.array(["NL" if "NL" in s else "LN" if "LN" in s else "PT" for s in plot_df.index])

    if ax is None:
        fig, ax = plt.subplots()

    ax.scatter(plot_df.iloc[:, 0].values, plot_df.iloc[:, 1].values, c=colors,
               marker='o', s=750, alpha=0.5)

    # ax.scatter(plot_df.iloc[:, 0].values[legend_text == "LN"], plot_df.iloc[:, 1].values[legend_text == "LN"],
    #            c=colors[legend_text == "LN"], marker='o', s=750, alpha=0.5)
    # ax.scatter(plot_df.iloc[:, 0].values[legend_text == "PT"], plot_df.iloc[:, 1].values[legend_text == "PT"],
    #            c=colors[legend_text == "PT"], marker='o', s=750, alpha=0.5)

    f1 = lambda l, c: plt.plot([], [], marker="o", color=c, ls="none", label=l)[0]

    handles = [f1(l, c) for l, c in [("NL", "tab:green"), ("LN", "tab:orange"), ("PT", "tab:blue")]]
    lgd = ax.legend(handles, labels, framealpha=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    ax.set_xlim([x_min, 1.1 * x_max])
    ax.set_ylim([y_min, 1.1 * y_max])

    counter = 0

    for i, j in zip(plot_df.iloc[:, 0].values, plot_df.iloc[:, 1].values):
        ax.annotate(labels[counter], xy=(i, j), color='k',
                    fontsize="large", weight='heavy',
                    horizontalalignment='center',
                    verticalalignment='center')

        counter += 1

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.legend(loc='best')

    if save_path is not None:
        plt.savefig(save_path)

    return ax


def compare_per_gene(exp_df1, exp_df2, gene="TP53", xlabel="Almac_expression", ylabel="Own expression"):
    common_samples = np.intersect1d(exp_df1.index.values, exp_df2.index.values)
    exp_df1, exp_df2 = exp_df1.loc[common_samples], exp_df2.loc[common_samples]

    colors = ["tab:orange" if "RP" in s else "tab:blue" if "NL" in s else "tab:green" for s in exp_df1.index]
    fig, ax = plt.subplots()
    ax.scatter(exp_df1[gene].values, exp_df2[gene].values, c=colors)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title("Expression for %s" % gene)
    plt.show()


def compare_per_patient(exp_df1, exp_df2, xlabel="Almac_expression",
                        ylabel="Own expression", sample="HR_ID9_NL"):

    common_genes = np.intersect1d(exp_df1.columns.values, exp_df2.columns.values)
    exp_df1, exp_df2 = exp_df1[common_genes], exp_df2[common_genes]

    fig, ax = plt.subplots()
    ax.scatter(exp_df1.loc[sample].values, exp_df2.loc[sample].values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.set_title("Expression for %s" % sample)
    plt.show()


def sns_scatter_with_contour(plot_df, group_col=None, save_name=None,  kind="kde", **kwargs):
    plot_df = copy.deepcopy(plot_df)
    x, y = plot_df.columns.values[:2]

    if group_col is None:
        plot_df["Tissue type"] = ["PT" if "RP" in sid else "MLN" if "LN" in sid else "AN" for sid in plot_df.index]

        jp = sns.jointplot(data=plot_df, x=x, y=y, kind=kind, hue="Tissue type", hue_order=['MLN', 'PT', 'AN'],
                           **kwargs)
    else:
        hues = np.unique(plot_df[group_col].values.flatten())
        jp = sns.jointplot(data=plot_df, x=x, y=y, kind=kind, hue=group_col, hue_order=hues, **kwargs)

    CS = jp.ax_joint

    max_diam = {tuple(level._edgecolors[0]): 0 for level in CS.collections}

    for level in CS.collections:
        edge_color = tuple(level._edgecolors[0])
        print(level._edgecolors)
        for kp, path in reversed(list(enumerate(level.get_paths()))):

            verts = path.vertices  # (N,2)-shape array of contour line coordinates
            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
            print(diameter)

            if diameter > max_diam[edge_color]:
                max_diam[edge_color] = diameter

        #print("Max diameter %f" % max_diam)
        for kp, path in reversed(list(enumerate(level.get_paths()))):
            verts = path.vertices  # (N,2)-shape array of contour line coordinates
            diameter = np.max(verts.max(axis=0) - verts.min(axis=0))
            if diameter < max_diam[edge_color]:  # threshold to be refined for your actual dimensions!
                del (level.get_paths()[kp])  # no remove() for Path objects:(

    # this might be necessary on interactive sessions: redraw figure
    plt.gcf().canvas.draw()

    if group_col is None:
        tissue2color = {"PT": "tab:orange", "AN": "tab:green", "MLN": "tab:blue"}

    else:
        hues = list(hues)
        colors = ["tab:blue", "tab:orange", "tab:green"]
        tissue2color = {h: c for h, c in zip(hues, colors)}

    markers = ["o", 's', 'x', "<", ">", "P", "^", "H", "+", "1", "2", "3", "*", "d", "H", "X", "D"]

    patients = np.unique(np.array([s.split("_")[1] for s in plot_df.index.values]))
    pat2marker = dict(zip(patients, markers))
    #pat2marker = {pat: "$" + pat[2:] + "$" for pat in patients}

    for idx, arr in plot_df.iterrows():
        pat = idx.split("_")[1]
        marker = pat2marker[pat]

        tissue = arr[-1]
        color = tissue2color[tissue]

        CS.scatter(arr[0], arr[1], c=color, marker=marker, s=30)

    if save_name is not None:
        im_path = FINAL_RESULTS_PATH + "/" + save_name + '.png'
        jp.savefig(im_path, dpi=600)


def sns_scatter_plot(plot_df, save_name=None,  kind="kde", **kwargs):
    plot_df = plot_df.copy(deep=True)
    x, y = plot_df.columns.values
    plot_df["Tissue type"] = ["RP" if "RP" in sid else "LN" if "LN" in sid else "AN" for sid in plot_df.index]
    plt.figure()
    jp = sns.jointplot(data=plot_df, x=x, y=y, kind=kind, hue="Tissue type", hue_order=['LN', 'RP', 'AN'],
                       **kwargs)

    if save_name is not None:
        im_path = RESULTS_PATH + "/figures/2d_scatter_plots/" + save_name + '.png'
        jp.savefig(im_path, dpi=600)


def plot_scatter_with_regression(plot_df, marker="o", color="tab:blue", ax=None, label=None,
                                 y_lim=None, x_lim=None):

    if ax is None:
        fig, ax = plt.subplots()

    x, y = plot_df.columns.values
    rho, pval = spearmanr(plot_df[x].values, plot_df[y].values)

    label = label + r" ($\rho={:10.2f}$)".format(rho)

    ax.scatter(plot_df[x].values, plot_df[y].values, marker=marker, c=color, label=label)

    m, b = np.polyfit(plot_df[x].values, plot_df[y].values, 1)

    xs = np.linspace(x_lim[0], x_lim[1], num=1000)

    ax.plot(xs, m * xs + b, c=color)

    if x_lim is not None:
        ax.set_xlim(x_lim)

    if y_lim is not None:
        ax.set_ylim(y_lim)


def hm2color():
    return {'Cancer-immunology': "tab:orange",
            'Activating invasion and metastasis': "k",
            'Sustaining proliferative signaling and evading growth suppressors': "g",
            'Deregulation of cellular energetics':"tab:purple",
            'Inducing angiogenesis': "tab:red",
            'Genome instability': "b",
            'Resisting cell death': "tab:grey",
            'Enabling replicative immortality': "tab:blue"}


def short_name_hallmarks():
    return {'Cancer-immunology': "Immunology",
            'Activating invasion and metastasis': "Invasion",
            'Sustaining proliferative signaling and evading growth suppressors': "Proliferation",
            'Deregulation of cellular energetics': "Energetics",
            'Inducing angiogenesis': "Angiogenesis",
            'Genome instability': "Genome instability",
            'Resisting cell death': "Cell death",
            'Enabling replicative immortality': "Immortality"}


def get_hallmarks_ordered():
    return ["Sustaining proliferative signaling and evading growth suppressors",
            'Enabling replicative immortality',
            'Cancer-immunology',
            'Activating invasion and metastasis',
            'Inducing angiogenesis',
            'Genome instability',
            'Resisting cell death',
            'Deregulation of cellular energetics'
             ]


def doughnut_plot(plot_df, hallmark_plot=False, colors=None, savename=None):
    fig, ax = plt.subplots()
    short_names = plot_df.index.values

    if hallmark_plot:
        order = get_hallmarks_ordered()
        plot_df = plot_df.loc[order]

        short_name_dict = short_name_hallmarks()
        short_names = [short_name_dict[g] for g in plot_df.index]

        colors = [hm2color()[idx] for idx in plot_df.index]

    wedges, texts = ax.pie(plot_df.values, colors=colors,
                           wedgeprops=dict(width=0.5), startangle=90, counterclock=False)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(short_names[i] + " (n=%i)"%plot_df.values[i], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

    if savename is not None:
        plt.savefig(savename + ".png", bbox_inches="tight", dpi=600)

    plt.show()


def doughnut_plot_2_columns(plot_df, hallmark_plot=False, colors=None, savename=None):
    fig, ax = plt.subplots()
    short_names = plot_df.index.values

    if hallmark_plot:
        order = get_hallmarks_ordered()
        plot_df = plot_df.loc[order]

        short_name_dict = short_name_hallmarks()
        short_names = [short_name_dict[g] for g in plot_df.index]

        colors = [hm2color()[idx] for idx in plot_df.index]

    wedges, texts = ax.pie(plot_df.n_hallmarks.values, colors=colors,
                           wedgeprops=dict(width=0.5), startangle=90, counterclock=False)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(short_names[i] + "\n(n=%i)" % plot_df.n_genes.values[i],
                    xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment=horizontalalignment, **kw)

        clr = "w" if colors[i] == "k" else "k"
        clr = "k"
        ax.annotate(plot_df.n_hallmarks.values[i], color=clr,
                    xy=(x, y), xytext=(0.7 * x, 0.7 * y),
                    horizontalalignment="center", bbox=dict(fc="w", ec="none", alpha=0.5, boxstyle="round"))

    if savename is not None:
        plt.savefig(savename + ".png", bbox_inches="tight", dpi=600)

    plt.show()

#
# def plot_AUROC(X, y):
#     cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
#     classifier = LogisticRegression(solver="lbfgs", class_weight="balanced")
#
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     for i, (train, test) in enumerate(cv.split(X, y)):
#         classifier.fit(X[train], y[train])
#         viz = plot_roc_curve(classifier, X[test], y[test],
#                              name='ROC fold {}'.format(i),
#                              alpha=0.3, lw=1, ax=None)
#         interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#         interp_tpr[0] = 0.0
#         tprs.append(interp_tpr)
#         aucs.append(viz.roc_auc)
#
#         plt.close()
#
#     fig, ax = plt.subplots()
#     ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#             label='Chance', alpha=.8)
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     ax.plot(mean_fpr, mean_tpr, color='b',
#             label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#             lw=2, alpha=.8)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                     label=r'$\pm$ 1 std. dev.')
#
#     ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05])
#     ax.legend(loc="lower right")
#
#     ax.spines["right"].set_visible(False)
#     ax.spines["top"].set_visible(False)
#     ax.set_xlabel("1 - Specificity", fontsize=16)
#     ax.set_ylabel("Sensitivity", fontsize=16)
#
#     plt.show()


def plot_concentric_pies(plot_df, hallmark_plot=True, save_name=None):
    fig, ax = plt.subplots()
    short_names = plot_df.index.values

    if hallmark_plot:
        order = get_hallmarks_ordered()
        plot_df = plot_df.loc[order]

        short_name_dict = short_name_hallmarks()
        short_names = [short_name_dict[g] for g in plot_df.index]

        colors = [hm2color()[idx] for idx in plot_df.index]

    wedges, texts = ax.pie(plot_df.iloc[:, 1].values, colors=colors, radius=1.1,
                           wedgeprops=dict(width=0.3, edgecolor='w'), startangle=90, counterclock=False,# autopct="n=%i",
                           textprops=dict(color="w"))

    w2, t2 = ax.pie(plot_df.iloc[:, 0].values, colors=colors, radius=.7,
                    wedgeprops=dict(width=0.3, edgecolor='w'), startangle=90, counterclock=False,
                    textprops=dict(color="w"))

    _, _ = ax.pie([1.], colors=["navy"], radius=.78,
                    wedgeprops=dict(width=0.05, edgecolor='w'), startangle=90, counterclock=False)

    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    kw = dict(arrowprops=dict(arrowstyle="-"),
              bbox=bbox_props, zorder=0, va="center")

    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(short_names[i] + "\n (n=%i)" %plot_df.iloc[i, 1], xy=(x, y), xytext=(1.35 * np.sign(x), 1.4 * y),
                    horizontalalignment="center", **kw)

    for i, p in enumerate(w2):
        ang = (p.theta2 - p.theta1) / 2. + p.theta1
        y = np.sin(np.deg2rad(ang)) * 0.27
        x = np.cos(np.deg2rad(ang)) * 0.27

        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        ax.annotate("%i" %plot_df.iloc[i, 0], xy=(x, y),
                    horizontalalignment=horizontalalignment, c=p._facecolor)

    ax.annotate("%i" %plot_df.iloc[:, 0].sum(), xy=(0, 0),
                horizontalalignment="center", c="k", fontsize=16)

    if save_name is not None:
        plt.savefig("results/figures/" + save_name + ".png", bbox_inches="tight")

    plt.show()


def arrowed_spines(ax,
                   x_width_fraction=0.02,
                   x_height_fraction=0.02,
                   lw=None,
                   ohg=0,
                   locations=('bottom right', 'left up'),
                   **arrow_kwargs):
    """
    Add arrows to the requested spines
    Code originally sourced here: https://3diagramsperpage.wordpress.com/2014/05/25/arrowheads-for-axis-in-matplotlib/
    And interpreted here by @Julien Spronck: https://stackoverflow.com/a/33738359/1474448
    Then corrected and adapted by me for more general applications.
    :param ax: The axis being modified
    :param x_{height,width}_fraction: The fraction of the **x** axis range used for the arrow height and width
    :param lw: Linewidth. If not supplied, default behaviour is to use the value on the current left spine.
    :param ohg: Overhang fraction for the arrow.
    :param locations: Iterable of strings, each of which has the format "<spine> <direction>". These must be orthogonal
    (e.g. "left left" will result in an error). Can specify as many valid strings as required.
    :param arrow_kwargs: Passed to ax.arrow()
    :return: Dictionary of FancyArrow objects, keyed by the location strings.
    """
    # set/override some default plotting parameters if required
    arrow_kwargs.setdefault('overhang', ohg)
    arrow_kwargs.setdefault('clip_on', False)
    arrow_kwargs.update({'length_includes_head': True})

    # axis line width
    if lw is None:
        # FIXME: does this still work if the left spine has been deleted?
        lw = ax.spines['left'].get_linewidth()

    annots = {}

    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()

    # get width and height of axes object to compute
    # matching arrowhead length and width
    fig = ax.get_figure()
    dps = fig.dpi_scale_trans.inverted()
    bbox = ax.get_window_extent().transformed(dps)
    width, height = bbox.width, bbox.height

    # manual arrowhead width and length
    hw = x_width_fraction * (ymax - ymin)
    hl = x_height_fraction * (xmax - xmin)

    # compute matching arrowhead length and width
    yhw = hw / (ymax - ymin) * (xmax - xmin) * height / width
    yhl = hl / (xmax - xmin) * (ymax - ymin) * width / height

    # draw x and y axis
    for loc_str in locations:
        side, direction = loc_str.split(' ')
        assert side in {'top', 'bottom', 'left', 'right'}, "Unsupported side"
        assert direction in {'up', 'down', 'left', 'right'}, "Unsupported direction"

        if side in {'bottom', 'top'}:
            if direction in {'up', 'down'}:
                raise ValueError("Only left/right arrows supported on the bottom and top")

            dy = 0
            head_width = hw
            head_length = hl

            y = ymin if side == 'bottom' else ymax

            if direction == 'right':
                x = xmin
                dx = xmax - xmin
            else:
                x = xmax
                dx = xmin - xmax

        else:
            if direction in {'left', 'right'}:
                raise ValueError("Only up/downarrows supported on the left and right")
            dx = 0
            head_width = yhw
            head_length = yhl

            x = xmin if side == 'left' else xmax

            if direction == 'up':
                y = ymin
                dy = ymax - ymin
            else:
                y = ymax
                dy = ymin - ymax

        annots[loc_str] = ax.arrow(x, y, dx, dy, fc='k', ec='k', lw=lw,
                                   head_width=head_width, head_length=head_length, **arrow_kwargs)

    return annots


def plot_patient_heterogeneity(score_series, ax=None, tissue_types=None, sort_patients_axis=True,
                               random_offset_per_sample=True, offset_interval=0.2,
                               ylabel=None,
                               random_seed=42,
                               special_group=None,
                               save_path=None):
    np.random.seed(random_seed)

    # rc = {"xtick.direction" : "inout", "ytick.direction" : "inout",
    #        "xtick.major.size" : 5, "ytick.major.size" : 5,}
    # with plt.rc_context(rc):

    if special_group is None:
        special_group = []

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    if ylabel is None:
        ylabel = score_series.name

    viridis = cm.get_cmap('tab20', 17)
    marker_types = np.array(["o", 's', 'x', "<", ">", "P", "^", "H", "+"])
    viridis_list = viridis(range(17))

    if tissue_types is None:
        tissue_types = np.array(['PT' if 'RP' in s.split("_")[-1] else 'LN' if 'LN' in s.split("_")[-1] else 'AN'
                                 for s in score_series.index.values])

    patients = np.array([s.split("_")[1] for s in score_series.index.values])

    if sort_patients_axis:
        uniq_patients = sort_ids_by_patient_sample_type_number(score_series.index.values)

    else:
        uniq_patients = np.unique(patients)

    tissue2int = {t: i for i, t in enumerate(np.unique(tissue_types))}
    pat2int = {t: i for i, t in enumerate(uniq_patients)}

    colors = [viridis_list[pat2int[pat]] for pat in patients]
    markers = ["s", "o"]

    xs = [pat2int[pat] for pat in patients]
    offset_per_sample = 0

    for i, v in enumerate(score_series.values.flatten()):
        sample = score_series.index.values[i]
        if random_offset_per_sample:
            random_offset_per_sample = (np.random.random() - .5) * offset_interval / .5

        tissue = 'PT' if 'RP' in sample.split("_")[-1] else 'LN' if 'LN' in sample.split("_")[-1] else 'AN'
        tissue_int = tissue2int[tissue]

        if sample in special_group:
            ec = "k"

        else:
            ec = None

        ax.scatter(xs[i] + random_offset_per_sample, v, c=colors[i], marker=markers[tissue_int], edgecolors=ec)

    ax.set_xlabel("Patient ID", fontsize=16)
    ax.set_ylabel(ylabel=ylabel, fontsize=16)

    ax.set_xticks(np.arange(len(uniq_patients)))
    ax.set_xticklabels(uniq_patients)

    f1 = lambda m, c: plt.scatter([], [], marker=m, edgecolors=c, c="w")

    handles = [f1(m, "k") for m in markers]
    labels = np.unique(tissue_types)

    lgd = ax.legend(handles, labels, framealpha=1, loc="lower right", fontsize=15)

    top = ax.get_ylim()[-1]
    ax.set_ylim([-top, top])

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    arrowed_spines(ax, locations=['left up', "left down"])

    ax.spines["bottom"].set_visible(False)
    ax.axhline(0.0, c="k")

    ax.set_yticks([-top + top/3, top - top/3])
    ax.set_yticklabels(["Closer to LN-", "Closer to LN+"], rotation=90, va="center", fontsize=16)
    ax.tick_params(axis='y', width=0.)
    ax.set_ylabel("Molecular Score")
    #ax.yaxis.set_major_formatter(plt.NullFormatter())

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()


def sort_sample_ids(arr):
    patient_ids = np.array([int(''.join(c for c in s.split("_")[1] if c.isdigit())) for s in arr])
    sample_ids = np.array([int(''.join(c for c in s.split("_")[-1] if c.isdigit())) for s in arr])
    sort_idx = np.lexsort((sample_ids, patient_ids))

    return arr[sort_idx]


def get_patient2color():
    pat2col = {'ID1': (0.12156863, 0.46666667, 0.70588235, 1.),
               'ID2': (0.68235294, 0.78039216, 0.90980392, 1.),
               'ID3': (1., 0.49803922, 0.05490196, 1.),
               'ID4': (1., 0.73333333, 0.47058824, 1.),
               'ID5': (0.59607843, 0.8745098, 0.54117647, 1.),
               'ID6': (0.83921569, 0.15294118, 0.15686275, 1.),
               'ID7': (1., 0.59607843, 0.58823529, 1.),
               'ID9': (0.58039216, 0.40392157, 0.74117647, 1.),
               'ID10': (0.54901961, 0.3372549 , 0.29411765, 1.),
               'ID11': (0.76862745, 0.61176471, 0.58039216, 1.),
               'ID15': (0.89019608, 0.46666667, 0.76078431, 1.),
               'ID16': (0.96862745, 0.71372549, 0.82352941, 1.),
               'ID17': (0.78039216, 0.78039216, 0.78039216, 1.),
               'ID18': (0.7372549 , 0.74117647, 0.13333333, 1.),
               'ID19': (0.85882353, 0.85882353, 0.55294118, 1.),
               'ID24': (0.09019608, 0.74509804, 0.81176471, 1.),
               'ID25': (0.61960784, 0.85490196, 0.89803922, 1.)}

    return pat2col