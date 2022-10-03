import sys
sys.path.append("../")
import pandas as pd
import numpy as np
import gzip
from utils.get_paths import INPUT_PATH, FINAL_RESULTS_PATH
from scipy.stats import fisher_exact
from joblib import Parallel, delayed


def perform_variant_testing(pileup_file, min_support,
                            normal_variants=None,
                            rnai_variants=None,
                            file_name="RP_2_MLN_pvals",
                            sample_file=INPUT_PATH + "tumor_files.txt",
                            save_path=FINAL_RESULTS_PATH,
                            chunksize=100_000,
                            debug_mode=False,
                            save_info=True):

    min_depth = min_support
    print("Starting analysis at depth %i" % min_depth)

    alt_cols, ref_cols = None, None
    with gzip.open(pileup_file, "rt") as f:
        line = f.readline()
        cols = line.split(",")

        ref_cols = [s for s in cols if s[-1] == "R"]
        alt_cols = [s for s in cols if s[-1] == "A"]

    valid_cols = ["Chromosome", "Ref", "Alt", "Position"] + ref_cols + alt_cols
    print("Number of samples detected: %i" % len(ref_cols))

    # if provided create a sample map, this used to identify variants that occur in multiple patients and might thus be
    # RNAi or germline events.
    if sample_file is not None:
        with open(sample_file) as f:
            line = f.readline()
            files = line.split(" ")
            samples = ["_".join(f.split("/")[-1].split(".")[0].split("_")[:3]) for f in files]

        file2sample = {"File%i" % (i + 1): sid for i, sid in enumerate(samples)}

    # Read in the data per chunks
    reader = pd.read_csv(pileup_file, index_col=None, usecols=valid_cols, chunksize=chunksize, header=0)

    t_overlap, valid_positions, n_variants = None, None, None

    for i, chunk in enumerate(reader):
        # chunk = reader.get_chunk()
        if not (debug_mode and (i >= 1)):
            chunk.index = chunk["Chromosome"].astype(str) + ":" + chunk["Position"].astype(str) + ":" + \
                          chunk["Ref"].astype(str)

            chunk = chunk.drop(columns=["Chromosome", "Ref", "Alt", "Position"])

            if rnai_variants is not None:
                chunk = chunk.loc[~chunk.index.isin(rnai_variants)]

            if normal_variants is not None:
                chunk = chunk.loc[~chunk.index.isin(normal_variants)]

            t_reads = chunk[alt_cols].T

            if sample_file is not None:
                t_reads.index = [file2sample[s[:-1]] for s in t_reads.index]

                var_present = 1 * (t_reads > min_support)
                patients = [s.split("_")[1] for s in t_reads.index]
                sums_per_pat = (var_present.groupby(patients).sum() > 0).sum(axis=0)
                valid_variants = sums_per_pat.index.values[sums_per_pat.values <= 2]

                chunk = chunk.loc[valid_variants]
                var_present = var_present[valid_variants]
                t_reads = t_reads[valid_variants]

            else:
                var_present = 1 * (t_reads > min_support)
                t_reads.index = [s[:-1] for s in t_reads.index]
                var_present.index = [s[:-1] for s in var_present.index]

            file_ids = [s[:-1] for s in chunk.columns]

            depth = chunk.transpose().groupby(file_ids, sort=False).sum()

            if sample_file is not None:
                depth.index = [file2sample[s] for s in depth.index]

            valid_mask = 1 * ((depth > min_support) &
                              ((t_reads > min_support) | (t_reads < 1)))

            # valid_mask = 1 * (depth > min_support)
            if valid_positions is None:
                valid_positions = valid_mask.dot(valid_mask.transpose())
            else:
                valid_positions += valid_mask.dot(valid_mask.transpose())

            if t_overlap is None:
                t_overlap = var_present.dot(var_present.T)

            else:
                t_overlap += var_present.dot(var_present.T)

            if n_variants is None:
                n_variants = valid_mask.dot(var_present.T)

            else:
                n_variants += valid_mask.dot(var_present.T)

    # Then we perform statistical testing to see if some samples show significant overlap.
    # we use fisher exact test to compare between patients
    #                Pat 1
    #           mut         no mut
    #       mut
    # Pat 2
    #    no mut

    if sample_file is None:
        all_rp_files = n_variants.index.values
        all_ln_files = n_variants.index.values

    else:
        all_rp_files = [s for s in n_variants.index.values if "RP" in s]
        all_ln_files = [s for s in n_variants.index.values if "MLN" in s]

    cont_table = np.zeros((2, 2))
    pvals = np.zeros((len(all_rp_files), len(all_ln_files)))

    if debug_mode:
        print(t_overlap)
        print(n_variants)
        print(valid_positions)

    for i, s1 in enumerate(all_rp_files):
        for j, s2 in enumerate(all_ln_files):
            n_common = t_overlap.loc[s1, s2]
            cont_table[0, 0] = n_common
            cont_table[1, 0] = n_variants.loc[s2, s1] - n_common
            cont_table[0, 1] = n_variants.loc[s1, s2] - n_common

            cont_table[1, 1] = valid_positions.loc[s1, s2] - cont_table[0, 0] - cont_table[1, 0] - cont_table[0, 1]

            S, pval = fisher_exact(cont_table)
            pvals[i, j] = pval

    pval_df = pd.DataFrame(pvals, index=all_rp_files, columns=all_ln_files)
    if file_name is not None:
        pval_df.to_csv(save_path + "/" + file_name + "_depth%i.csv" % min_depth)

    if save_info:
        t_overlap.to_csv(save_path + "/overlaps_depth%i.csv" % min_depth)
        valid_positions.to_csv(save_path + "/valid_positions_depth%i.csv" % min_depth)
        n_variants.to_csv(save_path + "/n_variants_depth%i.csv" % min_depth)

    return pval_df


if __name__ == "__main__":
    min_supports = np.arange(0, 21)
    n_jobs = 1
    # file to process snp-pileups for variant identification
    pileup_file = INPUT_PATH + "COSMIC/pileups_tumor.txt.gz"
    # perform_variant_testing(pileup_file, min_support=5, file_name=None, chunksize=1000, debug_mode=True)

    # first we cross-check with RADAR to remove variants that may be due to RNA-editing
    rnai_variants = pd.read_csv(INPUT_PATH + "COSMIC/TABLE1_hg38.txt.gz",
                                 sep="\t")

    rnai_variants = rnai_variants["Region"].astype(str) + ":" + \
                    rnai_variants["Position"].astype(str) + ":" + \
                    rnai_variants["Ref"].astype(str)

    rnai_variants = set(rnai_variants.to_list())

    # Create panel of normals:
    # We also create a panel of normals:
    normal_pileups = INPUT_PATH + "COSMIC/pileups_normal.txt.gz"
    valid_cols_norm = ["Chromosome", "Ref", "Alt", "Position"] + ["File%iA" % i for i in range(1, 18)]

    reader = pd.read_csv(normal_pileups, index_col=None, usecols=valid_cols_norm, chunksize=10000, header=0)
    normal_variants = []
    print("Processing normal variants...")

    for i, chunk in enumerate(reader):
        chunk.index = chunk["Chromosome"].astype(str) + ":" + chunk["Position"].astype(str) + ":" + \
                      chunk["Ref"].astype(str)
        chunk = chunk.drop(columns=["Chromosome", "Ref", "Alt", "Position"])
        mask = (chunk.values >= 2).sum(axis=1) > 0
        normal_variants += list(chunk.index.values[mask])

    normal_variants = set(normal_variants)

    if n_jobs == 1:
        for min_support in [2]:
            perform_variant_testing(pileup_file, min_support)

    else:
        Parallel(n_jobs=n_jobs, backend='loky')(delayed(perform_variant_testing)(pileup_file=pileup_file,
                                                                                 min_support=min_support,
                                                                                 normal_variants=normal_variants,
                                                                                 rnai_variants=rnai_variants,
                                                                                save_path=FINAL_RESULTS_PATH + "variant_calling_noCT"
                                                                                 )
                                                                  for min_support in min_supports)
