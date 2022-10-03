import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, rankdata
from sklearn.model_selection import RepeatedStratifiedKFold
import time
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA


class CentroidClassifier:

    def __init__(self, centroids=None, classes=None):
        self.centroids = centroids
        self.shrinkages = None
        self.classes = classes
        self.feature_names = None

    def check_init(self):
        assert self.centroids is not None, "Please initialize the centroids first using the fit function."

    def fit(self, X_train, y_train, aggr_func=np.mean, shrinkage_factor=None):

        if isinstance(X_train, pd.DataFrame):
            self.feature_names = X_train.columns.values
            X_train = X_train.values

        else:
            self.feature_names = np.arange(X_train.shape[1])

        if shrinkage_factor is None:
            self.classes = np.unique(y_train)
            self.centroids = np.asarray([aggr_func(X_train[y_train == l], axis=0) for l in self.classes])

        else:
            self.centroids, self.shrinkages = get_shrunken_centroids(X_train, y_train, shrinkage_factor=shrinkage_factor)

    def predict(self, X, dist_func="spearman-mat"):
        probs = self.predict_proba(X, dist_func=dist_func)
        return np.argmax(probs, axis=1)

    def fit_CV(self, X, y, metric_func, grid=np.arange(start=0, stop=1, step=0.05), n_splits=5, n_repeats=2):
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

        scores = {i: [] for i in grid}

        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            for shr in grid:
                clf = CentroidClassifier()
                clf.fit(X_train, y_train, shrinkage_factor=shr)
                probs = clf.predict_proba(X_test)

                scores[shr].append(metric_func(y_test, probs[:, 1]))

        scores = pd.DataFrame(scores)
        median_scores = scores.median(axis=0)
        best_skr = median_scores.idxmax()

        print("Finished Cross-Validation and selected an optimal shrinkage factor of %f." % best_skr)

        self.fit(X, y, shrinkage_factor=best_skr)

        return scores

    def predict_proba(self, X, dist_func="spearman-mat", norm="own"):
        if isinstance(dist_func, str):
            try:
                dist_func = str2func[dist_func]
            except KeyError:
                raise IOError("The input dist_func is not implemented yet. Please provide a function or 'spearman'/'pearson'.")

        preds = dist_func(X, self.centroids)

        if norm.lower() == "pam50":
            preds[preds < 0.5] = 0

        return normalize_distances(preds)

    @property
    def centroid_df(self):
        return pd.DataFrame(self.centroids.transpose(), index=self.feature_names, columns=self.classes)

    def plot_centroids(self, gene_list=None, **kwargs):
        plot_centroids(self.centroid_df, gene_list=gene_list, **kwargs)


def normalize_distances(dists):
    n_classes = dists.shape[1]
    scaling_factor = dists.sum(axis=1, keepdims=True)
    mask = scaling_factor < 1e-10
    scaling_factor[mask] = 1.
    dists /= scaling_factor
    dists[mask.flatten()] = 1./n_classes
    return dists


def test_normalize_distances():
    test_data1 = np.array([[0.4, 0.7, 0.5],
                           [0.4, 0.4, 0.4],
                           [0, 0, 0]])
    print(normalize_distances(test_data1))


class SpearmanPCA():
    """"
    Assumes X is a sample x gene DF"""
    def __init__(self, n_components=2, **kwargs):
        self.pca = PCA(n_components=n_components, **kwargs)
        self.X = None

    def fit(self, X):
        self.X = np.asarray(X)
        X_ranks = get_ranks_rowwise(X)
        self.pca.fit(X_ranks)

    def fit_transform(self, X):
        self.X = np.asarray(X)
        X_ranks = get_ranks_rowwise(X)
        return self.pca.fit_transform(X_ranks)

    def transform(self, X):
        new_ranks = get_ranks_rowwise(X)
        return self.pca.transform(new_ranks)


def get_original_rank(X, X_new):
    '''
    Calculate the rank a new datapoint would have on the data
    :param X:
    :param X_new:
    :return:
    '''

    ranks = []
    for v in X_new:
        X_tot = np.vstack((X, v[None, ...]))
        rank_ = get_ranks_rowwise(np.transpose(X_tot))[:, -1]
        ranks.append(rank_)

    return np.array(ranks)


def spearman_dist_mat(mat1, mat2):
    return (1. + spearman_mat(mat1, mat2)) / 2.


def pearson_dist_mat(mat1, mat2):
    return (1. + pearson_mat(mat1, mat2)) / 2.


def spearman_dist(v1, v2):
    return (1. + spearmanr(v1, v2)[0]) / 2.


def pearson_dist(v1, v2):
    return (1. + pearsonr(v1, v2)[0]) / 2.


str2func = {"spearman": spearman_dist, "pearson": pearson_dist, "spearman-mat": spearman_dist_mat,
            "pearson-mat": pearson_dist_mat}


def get_ranks_rowwise(arr):
    return np.asarray([rankdata(arr_) for arr_ in arr])


def spearman_mat(A, B):
    if len(A.shape) == 1:
        A = A[None, ...]

    if len(B.shape) == 1:
        B = B[None, ...]

    rA, rB = get_ranks_rowwise(A), get_ranks_rowwise(B)
    # Rowwise mean of input arrays & subtract from input arrays themselves
    A_mA = rA - rA.mean(1)[:, None]
    B_mB = rB - rB.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    denominator = np.sqrt(np.dot(ssA[:, None], ssB[None]))
    denominator[denominator < 1e-10] = 1.
    corr_coefs = np.dot(A_mA, B_mB.T) / denominator

    return corr_coefs


def pearson_mat(A, B):
    if len(A.shape) == 1:
        A = A[None, ...]

    if len(B.shape) == 1:
        B = B[None, ...]

    # Rowwise mean of input arrays & subtract from input arrays themselves
    A_mA = A - A.mean(1)[:, None]
    B_mB = B - B.mean(1)[:, None]

    # Sum of squares across rows
    ssA = (A_mA**2).sum(1)
    ssB = (B_mB**2).sum(1)

    # Finally get corr coeff
    return np.dot(A_mA, B_mB.T) / np.sqrt(np.dot(ssA[:, None], ssB[None]))


def test_corrmat():
    testmat1 = np.array([[1, 2, 3, 4],
                         [4, 5, 6, 7],
                         [8, 5, 3, 5]])

    testmat2 = np.array([[7, 2, 3, 4],
                         [3, 5, 9, 7],
                         [1, 5, 10, 5]])

    corrmat = np.zeros((testmat1.shape[0], testmat2.shape[0]))

    for i, v1 in enumerate(testmat1):
        for j, v2 in enumerate(testmat2):
            corrmat[i, j] = spearmanr(v1, v2)[0]

    corrmat2 = spearman_mat(testmat1, testmat2)
    assert np.all(np.abs(corrmat - corrmat2) < 1e-5)

    corr_vec = spearman_mat(testmat1.flatten(), testmat2.flatten())
    rho = spearmanr(testmat1.flatten(), testmat2.flatten())[0]
    assert np.all(np.abs(rho - corr_vec) < 1e-5)


def test_corrmat_pearson():
    testmat1 = np.array([[1, 2, 3, 4],
                         [4, 5, 6, 7],
                         [8, 5, 3, 5]])

    testmat2 = np.array([[7, 2, 3, 4],
                         [3, 5, 9, 7],
                         [1, 5, 10, 5]])

    corrmat = np.zeros((testmat1.shape[0], testmat2.shape[0]))

    for i, v1 in enumerate(testmat1):
        for j, v2 in enumerate(testmat2):
            corrmat[i, j] = pearsonr(v1, v2)[0]

    corrmat2 = pearson_mat(testmat1, testmat2)
    assert np.all(np.abs(corrmat - corrmat2) < 1e-5)

    corr_vec = pearson_mat(testmat1.flatten(), testmat2.flatten())
    rho = pearsonr(testmat1.flatten(), testmat2.flatten())[0]
    assert np.all(np.abs(rho - corr_vec) < 1e-5)


def within_class_variance(X, y):
    X = np.asarray(X)
    y = np.asarray(y)

    uniq_classes = np.unique(y)
    vars = np.zeros(X.shape[1])

    for cl in uniq_classes:
        X_k = X[y == cl]
        vars += np.sum((X_k - X_k.mean(axis=0, keepdims=True)) ** 2, axis=0)

    return vars/(X.shape[0] - len(uniq_classes))


def get_shrunken_centroids(X, y, shrinkage_factor=0.):

    X = np.asarray(X)
    y = np.asarray(y)

    uniq_classes = np.unique(y)
    means = X.mean(axis=0, keepdims=True)
    ds = np.zeros((len(uniq_classes), X.shape[1]))
    centroids = np.zeros((len(uniq_classes), X.shape[1]))
    denom = np.zeros((len(uniq_classes), 1))

    for i, cl in enumerate(uniq_classes):
        mask = y == cl
        centroids[i] = X[mask].mean(axis=0)
        denom[i] = np.sqrt(1/mask.sum() + 1/X.shape[0])
        ds[i] = (centroids[i] - means)/denom[i]

    s_i = within_class_variance(X, y)
    s0 = np.median(s_i)

    ds = ds/(s_i + s0)
    signs = np.sign(ds)
    shrink = np.maximum(np.abs(ds) - shrinkage_factor, 0)
    shrink *= signs

    centroids = means + shrink*(s_i + s0) * denom

    return centroids, shrink


def test_predict_proba():
    X_train = np.array([[1, 2, 3, 4],
                         [4, 5, 6, 7],
                         [8, 5, 3, 5]])

    y_train = np.array([0, 1, 1])

    X_test = np.array([[7, 2, 3, 4],
                       [3, 5, 9, 7],
                       [1, 5, 10, 5]])

    clf = CentroidClassifier()
    clf.fit(X_train, y_train)

    start = time.time()
    preds1 = clf.predict_proba(X_test)
    end = time.time()
    print("Elapsed time: %f" % (end - start))

    start = time.time()
    preds2 = clf.predict_proba(X_test)
    end = time.time()
    print("Elapsed time: %f" % (end - start))

    print(preds1)
    print(preds2)

    assert np.all(np.abs(preds1 - preds2) < 1e-5)

    clf = CentroidClassifier()
    clf.fit(X_train, y_train, shrinkage_factor=0.5)


def plot_centroids(centroid_df, gene_list=None, sort_by=None):

    centroid_df = centroid_df.copy(deep=True)

    if gene_list is not None:
        gene_list = np.asarray(gene_list)
        common_genes = np.intersect1d(gene_list, centroid_df.index.values)

        if len(common_genes) == 0:
            raise IOError("None of the genes provided are present in the centroids.")

    if sort_by is not None:
        centroid_df = centroid_df.sort_values(by=sort_by)

    ax = centroid_df.plot.bar(rot=0)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    new_patches = []

    for patch in reversed(ax.patches):
        bb = patch.get_bbox()
        color = patch.get_facecolor()
        p_bbox = FancyBboxPatch((bb.xmin, bb.ymin),
                                abs(bb.width), abs(bb.height),
                                boxstyle="round,pad=-0.0040,rounding_size=0.015",
                                ec="none", fc=color,
                                mutation_aspect=4
                                )
        patch.remove()
        new_patches.append(p_bbox)
    for patch in new_patches:
        ax.add_patch(patch)