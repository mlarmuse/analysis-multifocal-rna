import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata


class SeedsTargetsMatrix():

    def __init__(self, df):
        self.df = df

    @classmethod
    def from_file(cls, path, **kwargs):
        df = pd.read_csv(path, index_col=0, **kwargs)
        return cls(df)

    @property
    def target_samples(self):
        return self.df.columns.values

    @property
    def source_samples(self):
        return self.df.index.values

    @property
    def source_patient_ids(self):
        return np.asarray([s.split("_")[1] for s in self.source_samples])

    @property
    def target_patient_ids(self):
        return np.asarray([s.split("_")[1] for s in self.target_samples])

    @property
    def patients(self):
        return np.union1d(self.target_patient_ids, self.source_patient_ids)

    def subset_types(self, seed_type=None, target_type=None):

        df = self.df.copy(deep=True)
        if seed_type is not None:
            mask = [seed_type in s for s in self.df.index.values]
            df = df.loc[mask]

        if target_type is not None:
            mask = [target_type in s for s in self.df.columns.values]
            df = df.loc[:, mask]

        return SeedsTargetsMatrix(df)

    def get_matrix_per_patient(self):

        patients = self.patients
        target_pats, source_pats = self.target_patient_ids, self.source_patient_ids

        odict = {}

        for pat in patients:

            tmask, smask = target_pats == pat, source_pats == pat

            if (tmask.sum() > 0) and (smask.sum() > 0):
                odict[pat] = self.df.loc[smask].loc[:, tmask]

        return odict

    def get_seeding_samples(self):

        if len(self.patients) > 1:
            warnings.warn("There appears to be more than one patient id, "
                          "seeding samples do not have meaning when comparing across patients.")

        mat = self.df.values

        r, c = np.where(mat == np.min(mat, axis=0, keepdims=True))

        seeds = self.source_samples[r]
        targets = self.target_samples[c]

        odict = {s: [] for s in seeds}

        for s, t in zip(seeds, targets):
            odict[s] += [t]

        return odict

    def find_all_seeds(self):
        mats = self.get_matrix_per_patient()
        odict = {}

        for pat, mat in mats.items():
            pat_seeding_dict = SeedsTargetsMatrix(mat).get_seeding_samples()
            odict = {**odict, **pat_seeding_dict}

        return odict

    def get_min_values_per_target(self):
        mdict = self.get_matrix_per_patient()

        oseries = []

        for k, v in mdict.items():
            oseries.append(v.min(axis=0))

        return pd.concat(oseries, axis=0)

    def get_best_to_second_best_seeding_score(self):
        mdict = self.get_matrix_per_patient()

        oseries = []

        for k, v in mdict.items():
            mat = np.sort(v.values, axis=0)
            oseries.append(pd.Series(mat[-2, :]/mat[-1, :], index=v.columns))

        return pd.concat(oseries, axis=0)

    def get_rank_score(self):
        return get_rank_score(self.df)

    def get_best_to_second_best_rank_score(self):
        mdict = self.get_matrix_per_patient()

        oseries = {}

        for k, v in mdict.items():
            v, _ = get_rank_score(v)
            v_sort = np.sort(v.values.flatten())
            oseries[k] = v_sort[1]/v_sort[0]

        return pd.Series(oseries)

    def get_heterogeneity_score(self, aggr_func=np.mean):
        patients = self.patients
        target_pats, source_pats = self.target_patient_ids, self.source_patient_ids

        odict = {}

        for pat in patients:

            tmask, smask = target_pats == pat, source_pats == pat

            if (tmask.sum() > 0) and (smask.sum() > 0):
                self_values = aggr_func(calculate_overlapping_samples(self.df.loc[smask].loc[:, tmask]))
                other_values = aggr_func(self.df.loc[smask].loc[:, ~tmask].values.flatten())

                odict[pat] = other_values/self_values

        return odict


def get_ranks_column_wise(arr):
    return np.transpose(np.asarray([rankdata(arr[:, i]) for i in range(arr.shape[1])]))


def get_rank_score(df):
    c_rank_sums = get_ranks_column_wise(df.values).sum(axis=1)
    c_rank_sums = pd.Series(c_rank_sums, index=df.index.values)
    seed = c_rank_sums.idxmin()

    return c_rank_sums, seed


def calculate_overlapping_samples(df):

    common_samples = np.intersect1d(df.index.values, df.columns.values)

    mat = df.loc[common_samples, common_samples].values
    iu = np.triu_indices(len(common_samples), 1)

    mask = ~(np.isin(df.index.values, common_samples)[..., None] &
             np.isin(df.columns.values, common_samples)[None, ...])

    other_values = df.values[mask]

    return np.hstack((other_values, mat[iu]))


def test_calculate_overlapping_samples():
    testmat = np.array([[1, 2, 3, 4],
                        [2, 1, 2, 3],
                        [3, 2, 1, 2],
                        [4, 3, 2, 1]])

    ridx, cidx = ["S" + str(i) for i in range(4)], ["S" + str(i) for i in range(4)]
    test_df = pd.DataFrame(testmat, index=ridx, columns=cidx)
    print(calculate_overlapping_samples(test_df))

    cidx = ["S" + str(i) for i in range(3)] + ["S5"]
    test_df = pd.DataFrame(testmat, index=ridx, columns=cidx)
    print(calculate_overlapping_samples(test_df))

    cidx = ["S2", "S5", "S1", "S6"]
    test_df = pd.DataFrame(testmat, index=ridx, columns=cidx)
    print(len(calculate_overlapping_samples(test_df)))

    cidx = ["S2", "S5", "S8", "S6"]
    test_df = pd.DataFrame(testmat, index=ridx, columns=cidx)
    print(len(calculate_overlapping_samples(test_df)))


def test_get_heterogeneity_score():

    testmat = np.array([[1, 1, 0.6, 0.6],
                        [1, 1, 0.6, 0.6],
                        [0.2, 0.2, 1, 1],
                        [0.2, 0.2, 1, 1]])

    ridx, cidx = ["HR_ID1_RP1", "HR_ID1_RP2", "HR_ID2_RP1", "HR_ID2_RP2"],\
                 ["HR_ID1_MLN1", "HR_ID1_MLN2", "HR_ID2_MLN1", "HR_ID2_MLN2"]
    test_df = pd.DataFrame(testmat, index=ridx, columns=cidx)
    ds = SeedsTargetsMatrix(test_df).get_heterogeneity_score(np.mean)
    print(ds)

    ridx, cidx = ["HR_ID1_RP1", "HR_ID1_RP2", "HR_ID2_RP1", "HR_ID2_RP2"], \
                 ["HR_ID1_RP1", "HR_ID1_RP2", "HR_ID2_RP1", "HR_ID2_RP2"]
    test_df = pd.DataFrame(testmat, index=ridx, columns=cidx)
    ds = SeedsTargetsMatrix(test_df).get_heterogeneity_score(np.mean)
    print(ds)