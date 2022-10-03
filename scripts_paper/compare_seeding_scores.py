import sys
sys.path.append("../")

import pandas as pd
import numpy as np
from utils.statistics import poisson_binom_pmf
from utils.read_in_utils import get_samples_of_sufficient_purity, get_confirmed_patients
from scipy.stats import mannwhitneyu
from utils.SeedsTargetsMatrix import SeedsTargetsMatrix
from utils.get_paths import FINAL_RESULTS_PATH, INPUT_PATH


def identify_seeding_clone_per_MLN(dist_mat, pval_thresh=1e-10):
    """
    Strategy one: take the samples that is closest to all MLN samples, if such samples exists
    :param dist_mat:
    :return: the seeding samples and a flag if this seeding sample is the same for all MLN samples
    """
    single_seeding_sample = False
    remove_mask = dist_mat.min(axis=0).values < pval_thresh

    dist_mat = dist_mat.loc[:, remove_mask]
    seeding_samples = dist_mat.idxmin(axis=0)

    if len(pd.unique(seeding_samples)) == 1:
        single_seeding_sample = True

    return seeding_samples.values, single_seeding_sample


def score_seeding_samples(dist_mat, pval_thresh=1e-10):
    """
    Strategy 2: take the score across all different MLN samples and calculate the global score for each sample.
    :param dist_mat:
    :param pval_thresh:
    :return:
    """
    remove_mask = dist_mat.min(axis=0).values < pval_thresh
    dist_mat = dist_mat.loc[:, remove_mask]

    dist_mat = -10 * np.log10(dist_mat)
    scores = dist_mat/np.sum(dist_mat.values, axis=0, keepdims=True)
    scores = scores.sum(axis=1)
    return scores, scores.idxmax()


if __name__ == "__main__":
    depth = 6
    confirmed_patients = True
    tag = ""

    fraction_correct = {}
    pval_per_patient = {}

    pure_samples = get_samples_of_sufficient_purity()
    signature_scores = pd.read_csv(FINAL_RESULTS_PATH + "different_seed_scoring_schemes_signatures.csv",
                                   index_col=0)
    variant_scores = pd.read_csv(INPUT_PATH + "RP_2_MLN_pvals_clear_depth6.csv", index_col=0)
    variant_scores = variant_scores[[s for s in pure_samples if "MLN" in s]]

    if confirmed_patients:
        tag = "_confirmed_patients"
        confirmed_patients = get_confirmed_patients()
        signature_scores = signature_scores.loc[[s.split("_")[1] in confirmed_patients for s in signature_scores.index.values]]
        variant_scores = variant_scores.loc[[s.split("_")[1] in confirmed_patients for s in variant_scores.index.values]]

    r_patient_ids = np.array([s.split("_")[1] for s in variant_scores.index.values])
    c_patient_ids = np.array([s.split("_")[1] for s in variant_scores.columns.values])
    patients = np.array([s.split("_")[1] for s in signature_scores.index.values])

    uniq_patients = np.unique(r_patient_ids)
    seeds_signatures = []
    seeds_vars, strict_seeds = [], []
    score_per_sample = []
    lesion_probs, lesion_probs_mono, lesion_probs_poly = [], [], []
    overlap_lesion, overlap_lesion_mono, overlap_lesion_poly = 0, 0, 0
    count_mono, count_poly = 0, 0
    n_mlns_mono, n_mlns_poly = 0, 0
    mono_pats = []
    pat_count, overlap_count = 0, 0
    pat_probs = []
    confirmed_pats = []
    ranks_seed_sign = {}

    for pat in uniq_patients:
        pat_vars = variant_scores.loc[r_patient_ids == pat].loc[:, c_patient_ids == pat]

        if pat_vars.shape[1] > 0:
            scores, seeds = score_seeding_samples(pat_vars)
            score_per_sample.append(scores)
            seeds_vars.append(seeds)

            seeds_per_lesion, flag = identify_seeding_clone_per_MLN(pat_vars)

            sign_seed = signature_scores["Votes"].loc[pat == patients].idxmax()

            if flag:
                strict_seeds.append(seeds_per_lesion[0])
                #print(pat_vars)
            else:

                print("polyclonal seeding")
                print(sign_seed)
                print(pat_vars)

            seeds_signatures.append(sign_seed)

            overlap_lesion += np.isin(seeds_per_lesion, sign_seed).sum()

            n_rps, n_mlns = pat_vars.shape
            n_mlns = len(seeds_per_lesion)
            lesion_probs += ([1./n_rps] * n_mlns)

            if flag:
                overlap_lesion_mono += np.isin(seeds_per_lesion, sign_seed).sum()
                n_mlns_mono += n_mlns
                lesion_probs_mono += ([1./n_rps] * n_mlns)
                count_mono += 1
                mono_pats.append(pat)

            else:
                overlap_lesion_poly += np.isin(seeds_per_lesion, sign_seed).sum()
                n_mlns_poly += n_mlns
                lesion_probs_poly += ([1./n_rps] * n_mlns)
                count_poly += 1

            most_seeding_var, seed_var = SeedsTargetsMatrix(pat_vars).get_rank_score()
            var_rank_per_lesion = most_seeding_var.rank()
            ranks_seed_sign[pat] = var_rank_per_lesion.loc[sign_seed]

            if var_rank_per_lesion.loc[seed_var] <= 2:
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!Irregular behaviour found for:")
                print(pat)

            pat_count += 1
            #overlap_count += (sign_seed == seed_var)

            if sign_seed == seed_var:
                overlap_count += 1
                confirmed_pats.append(pat)

            pat_probs += [1./n_rps]

    fraction_correct[depth] = overlap_count / pat_count
    pval_per_patient[depth] = poisson_binom_pmf(pat_probs)[overlap_count:].sum()

    pval_all = poisson_binom_pmf(lesion_probs)[overlap_lesion:].sum()
    pval_mono = poisson_binom_pmf(lesion_probs_mono)[overlap_lesion_mono:].sum()
    pval_poly = poisson_binom_pmf(lesion_probs_poly)[overlap_lesion_poly:].sum()

    print("Depth: %i" % depth)
    print("P-value monoclonal seeding: %f (n=%i; %i out of %i lesions correct)" % (pval_mono, count_mono, overlap_lesion_mono, n_mlns_mono))
    print("P-value polyclonal seeding: %f (n=%i; %i out of %i lesions correct)" % (pval_poly, count_poly, overlap_lesion_poly, n_mlns_poly))
    print("P-value all: %f (n=%i; %i out of %i lesions correct)" % (pval_all, count_mono + count_poly,
                                                                    overlap_lesion_mono + overlap_lesion_poly,
                                                                    n_mlns_poly + n_mlns_mono))

    score_per_sample = pd.concat(score_per_sample, axis=0)
    score_per_sample.name = "Varscore"

    common_seeds = np.intersect1d(seeds_signatures, seeds_vars)
    common_seeds_strict = np.intersect1d(seeds_signatures, strict_seeds)

    # 8 seeds out of 17 are overlapping
    all_data = pd.merge(signature_scores, score_per_sample, left_index=True, right_index=True)

    # check if 6 out of 8 is significant

    patients_with_seed = [s.split("_")[1] for s in strict_seeds]
    patients_with_seed, counts = np.unique(r_patient_ids[np.isin(r_patient_ids, patients_with_seed)],
                                           return_counts=True)
    pval = poisson_binom_pmf(1./counts)[len(common_seeds_strict):].sum()

    print("P-value at patient level: %f" % pval)

    # test if monoclonal seeds are more significant

    mono_pvals, poly_pvals = [], []
    for pat in uniq_patients:
        pat_vars = variant_scores.loc[r_patient_ids == pat].loc[:, c_patient_ids == pat]

        min_vars = pat_vars.min(axis=0).values

        if pat in mono_pats:
            mono_pvals += list(min_vars)

        else:
            poly_pvals += list(min_vars)

    mannwhitneyu(mono_pvals, poly_pvals)

    np.median(mono_pvals)
    np.median(poly_pvals)

    # fractions_correct:

    pval = poisson_binom_pmf(pat_probs)[overlap_count:].sum()

    print("The p-value for the overlap between signature-based seeding lesion and somatic variant based seeding lesions"
          "is: %f" % pval)