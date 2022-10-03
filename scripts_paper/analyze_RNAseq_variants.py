import pandas as pd
import gzip
from utils.get_paths import INPUT_PATH, FINAL_RESULTS_PATH


def count_variant_signatures(pileup_file, min_support,
                            normal_variants=None, rnai_variants=None,
                            save_path=FINAL_RESULTS_PATH,
                            file_name="mut_signature_counts_",
                            file_name_cos="mut_per_patient_counts_",
                            sample_file=INPUT_PATH + "tumor_files.txt",
                            chunksize=100_000,
                            debug_mode=False):

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
    sample_counts = None
    var_sign_counts = []
    for i, chunk in enumerate(reader):
        # chunk = reader.get_chunk()
        if not (debug_mode and (i >= 1)):
            chunk.index = chunk["Chromosome"].astype(str) + ":" + chunk["Position"].astype(str) + ":" + \
                          chunk["Ref"].astype(str) + ":" + chunk["Alt"].astype(str)

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

                var_present = var_present[valid_variants]

            else:
                var_present = 1 * (t_reads > min_support)

        if sample_counts is None:
            sample_counts = var_present.sum(axis=1)

        else:
            sample_counts += var_present.sum(axis=1)

        var_sum = var_present.sum(axis=0)
        var_signature = [">".join(s.split(":")[-2:]) for s in var_sum.index.values]
        var_signature_df = var_sum.groupby(var_signature).sum()

        var_sign_counts += [var_signature_df]

    concat_df = pd.concat(var_sign_counts, axis=0)
    sum_series = concat_df.groupby(concat_df.index).sum().sort_values(ascending=False)

    if file_name is not None:
        sum_series.to_csv(save_path + "/" + file_name + "_depth%i.csv" % min_depth)

    if file_name_cos is not None:
        sample_counts.to_csv(save_path + "/" + file_name_cos + "_depth%i.csv" % min_depth)

    return var_sign_counts, sample_counts


def filter_pileup(pileup_file, chunksize=100_000, new_file=INPUT_PATH + "pileups_tumor.txt.gz"):
    reader = pd.read_csv(pileup_file, index_col=None, chunksize=chunksize, header=0)

    new_pileup = []
    for i, chunk in enumerate(reader):
        # chunk = reader.get_chunk()
        print(i)

        #chunk = chunk.loc[(chunk.Ref != "G") & (chunk.Alt != "A")]
        chunk = chunk.loc[(chunk.Ref != "C") & (chunk.Alt != "T")]

        new_pileup += [chunk]

    new_pileup = pd.concat(new_pileup, axis=0)
    new_pileup.to_csv(new_file, index=False, compression="gzip")


if __name__ == "__main__":

    # file to process snp-pileups for variant identification
    pileup_file = INPUT_PATH + "pileups_tumor.txt.gz"
    # perform_variant_testing(pileup_file, min_support=5, file_name=None, chunksize=1000, debug_mode=True)

    # first we cross-check with RADAR to remove variants that may be due to RNA-editing
    rnai_variants = pd.read_csv("TABLE1_hg38.txt.gz",
                                 sep="\t")

    rnai_variants = rnai_variants["Region"].astype(str) + ":" + \
                    rnai_variants["Position"].astype(str) + ":" + \
                    rnai_variants["Ref"].astype(str) + ":" + \
                    rnai_variants["Ed"].astype(str)

    rnai_variants = set(rnai_variants.to_list())

    # Create panel of normals:
    # We also create a panel of normals:
    normal_pileups = INPUT_PATH + "pileups_normal.txt.gz"
    valid_cols_norm = ["Chromosome", "Ref", "Alt", "Position"] + ["File%iA" % i for i in range(1, 18)]

    reader = pd.read_csv(normal_pileups, index_col=None, usecols=valid_cols_norm, chunksize=10000, header=0)
    normal_variants = []
    print("Processing normal variants...")

    for i, chunk in enumerate(reader):
        chunk.index = chunk["Chromosome"].astype(str) + ":" + chunk["Position"].astype(str) + ":" + \
                      chunk["Ref"].astype(str) + ":" + chunk["Alt"].astype(str)

        chunk = chunk.drop(columns=["Chromosome", "Ref", "Alt", "Position"])
        mask = (chunk.values >= 2).sum(axis=1) > 0
        normal_variants += list(chunk.index.values[mask])

    normal_variants = set(normal_variants)
    min_support = 6
    var_counts = count_variant_signatures(pileup_file,
                                          min_support=min_support,
                                          normal_variants=normal_variants,
                                          rnai_variants=rnai_variants,
                                          file_name="mut_signature_counts_%i" % min_support)
