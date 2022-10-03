import pandas as pd
import numpy as np
from utils.get_paths import INPUT_PATH, FINAL_RESULTS_PATH
from utils.misc import split_in_tissue_types, get_tissue_type
import matplotlib.pyplot as plt
import os
import gzip
from utils.SeedsTargetsMatrix import SeedsTargetsMatrix

plt.close("all")


def correct_signature_ids(signature_ids):
    invalid_ids_mask = np.array([len(s.split("_")) != 3 for s in signature_ids])
    invalid_ids = signature_ids[invalid_ids_mask]

    corr_signature_ids = [s if s not in invalid_ids else "HR_ID5_NL" if s == "ID5MLNL" else "HR_" + s
                          for s in signature_ids]
    assert all(len(s.split("_")) == 3 for s in corr_signature_ids)

    corr_signature_ids = [s if s.split("_")[0] == "HR" else "HR" + s[2:]
                          for s in corr_signature_ids]

    check_signature_ids(corr_signature_ids)

    return corr_signature_ids


def check_signature_ids(signature_ids, sample_types=("NL", "RP", "LN")):

    for s in signature_ids:
        hr_prefix = s.split("_")[0]
        assert hr_prefix == "HR"

        id = s.split("_")[1]
        assert id == check_id(id)

        specimen_id = s.split("_")[2]
        assert any(st in specimen_id for st in sample_types)


def read_in_signature_values(signatures_path, return_index2biology=False):
    signature_values = pd.read_csv(signatures_path, sep='\t')

    index = [sign if sign_name == "nan" else sign_name.split("[")[0].strip()
             for sign, sign_name in
             zip(signature_values.Signature.values, signature_values.Signature_as_named_in_report.values.astype(str))]

    signature_values.index = index

    index2biology = {index: biology for index, biology in zip(signature_values.index.values,
                                                              signature_values.Biology.values)}

    signature_values = signature_values.drop(columns=["Signature", "Signature_as_named_in_report", "Biology"])
    signature_values = signature_values.drop_duplicates()
    assert len(signature_values.index) == len(np.unique(signature_values.index))

    signature_values.columns = correct_signature_ids(signature_values.columns.values)

    if return_index2biology:
        return signature_values, index2biology

    else:
        return signature_values


def check_id(s):
    assert s[:2] == "ID"
    suffix = s[2:]

    integer = True
    i = 1

    while integer and (i <= len(suffix)):
        try:
            ID = int(suffix[:i])
            i += 1

        except ValueError:
            integer = False

    return "ID" + str(ID)


def read_in_stringtie_expression(normalization="TPM"):
    file = INPUT_PATH + "stringtie_exp_mat_" + normalization + ".csv.gz"
    df = pd.read_csv(file, index_col=0)

    df = df.rename(index={"HR_ID5_MLNL": "HR_ID5_NL"})
    #df = df.drop(columns=[BLACK_LIST_GENES])
    return df.transpose()


def read_in_own_signatures(file=INPUT_PATH+"cancer_hallmark_signatures.csv",
                           return_index2biology=True):
    signature_values = pd.read_csv(file, sep=",", index_col=1, header=None, comment="#")
    signature_values = signature_values.loc[[str(s) != "nan" for s in signature_values.index]]
    index2biology = dict(zip(signature_values.index.values, signature_values.iloc[:, 0]))

    signature_values = signature_values.iloc[:, 1:]
    signature_values = signature_values.applymap(lambda s: str(s).strip().upper())

    assert len(signature_values.index) == len(np.unique(signature_values.index))

    if return_index2biology:
        return signature_values, index2biology

    else:
        return signature_values


def get_expression(exp_df=None, normalization="TPM", return_tag=False,
                   filter_out_low_purity=True, purity_thresh=0.2, remove_zv_genes=True,
                   log_tf=True, collection_genes_to_keep=None, norm_patients=False, reference_tissue="NL",
                   calc_zscore=False):

    if exp_df is not None:
        exp_df = exp_df.copy()

    else:
        exp_df = read_in_stringtie_expression(normalization=normalization)
        tag = "_nextflow_stringtie"

    if collection_genes_to_keep is not None:
        collection_genes = get_signature_genes(collection_name=collection_genes_to_keep)
        common_genes = np.intersect1d(exp_df.index.values, collection_genes)
        print("There are %i genes in common." % len(common_genes))
        exp_df = exp_df.loc[common_genes]

    if filter_out_low_purity:
        purities = get_sample_purities()
        samples = purities.loc[purities > purity_thresh].index.values
        exp_df = exp_df[samples]

    if remove_zv_genes:
        exp_df = remove_tissue_specific_genes(exp_df)

    if log_tf:
        exp_df = np.log2(1. + exp_df)

    if calc_zscore:
        exp_df = get_zscore(exp_df, norm_patients=norm_patients, reference_tissue=reference_tissue)

    if return_tag:
        return exp_df, tag
    else:
        return exp_df


def get_samples_of_sufficient_purity(purity_thresh=0.2):
    purities = get_sample_purities()
    samples = purities.loc[purities > purity_thresh].index.values
    return samples


def get_sample_purities(path=INPUT_PATH,
                        file="EPIC_bref_cell_fractions_NextFlow.csv"):
    bref = pd.read_csv(path + file, index_col=0)
    bref = bref.rename(index={"HR_ID5_MLNL": "HR_ID5_NL"})
    return pd.Series(bref.iloc[:, -1], name="purities")


def get_infiltration_scores(path=INPUT_PATH,
                            file="EPIC_bref_cell_fractions_NextFlow.csv"):

    til_scores = pd.read_csv(path + file, index_col=0)
    til_scores = til_scores.rename(index={"HR_ID5_MLNL": "HR_ID5_NL"})
    return til_scores


def get_signatures(collection_name="own",
                   signature_path=INPUT_PATH,
                   return_index2biology=True):

    if "msig" in collection_name.lower():
        file_name = "msig_db_hallmarks.csv"
    elif "immune" in collection_name.lower():
        file_name = "immune_signatures.csv"
    elif ("pca" in collection_name.lower()) or ("prostate" in collection_name.lower()):
        file_name = "PCa_signatures_RB_NPEC.csv"
    else:
        file_name = "cancer_hallmark_signatures.csv"

    signature_df = pd.read_csv(signature_path + file_name,
                               comment="#",
                               engine="python")

    index2biology = dict(zip(signature_df["Signature name"], signature_df["Hallmark"]))
    signature_df = signature_df.iloc[:, 2:]
    signature_df = signature_df.set_index("Signature name")

    if return_index2biology:
        return signature_df, index2biology

    else:
        return signature_df


def get_signature_genes(collection_name="own", exp=None, min_ngenes=None):

    if "all" in collection_name.lower():
        signature_df = read_in_all_signatures(return_index2biology=False)

    else:
        signature_df = get_signatures(collection_name=collection_name,
                                      signature_path=INPUT_PATH,
                                      return_index2biology=False)

    if min_ngenes is not None:
        sign_sizes, _ = get_signature_sizes(exp=exp, signatures=signature_df)
        sign_sizes = pd.Series(sign_sizes)
        valid_signatures = sign_sizes.loc[sign_sizes >= min_ngenes].index
        signature_df = signature_df.loc[valid_signatures]


    genes = np.unique(signature_df.values.astype(str))

    return [s for s in genes if s != "nan"]


def get_genes_per_hallmark(collection_name="all", min_ngenes=None):
    if "all" in collection_name.lower():
        signature_df, index2biology = read_in_all_signatures(return_index2biology=True)

    else:
        signature_df, index2biology = get_signatures(collection_name=collection_name,
                                                     signature_path=INPUT_PATH,
                                                     return_index2biology=True)

    if min_ngenes is not None:
        sign_sizes, _ = get_signature_sizes(signatures=signature_df)
        sign_sizes = pd.Series(sign_sizes)
        valid_signatures = sign_sizes.loc[sign_sizes >= min_ngenes].index
        signature_df = signature_df.loc[valid_signatures]

    hallmark2genes = {index2biology[signature]: [] for signature in signature_df.index}
    for signature in signature_df.index:
        hallmark2genes[index2biology[signature]] += [g for g in signature_df.loc[signature].astype(str) if g != "nan"]

    hallmark2genes = {k: set(v) for k, v in hallmark2genes.items()}

    return hallmark2genes


#### for VCF data, no provided

def get_vcf_paths_for_patient(patient_id, conv_table,
                              vcf_path=INPUT_PATH + "Variant_calling/Platypus"):

    #sample_ids = conv_table.loc[[s.split("_")[1] == patient_id for s in conv_table.corr_sample_id]].file_id_corr
    sample_ids = [s for s in conv_table.corr_sample_id if s.split("_")[1] == patient_id ]

    return [os.path.join(vcf_path, sample_id + ".vcf") for sample_id in sample_ids]


def get_all_vcf_paths(conv_table,
                      vcf_path=INPUT_PATH + "Variant_calling/Platypus"):

    sample_ids = conv_table.corr_sample_id

    return [os.path.join(vcf_path, sample_id + ".vcf") for sample_id in sample_ids.values]


def read_in_vcf(vcf_path, filter=True):
    if vcf_path[-3:] == ".gz":
        with gzip.open(vcf_path, 'rt') as f:
            lines = [i for i, l in enumerate(f) if l.startswith("##")]

    else:
        with open(vcf_path, 'r') as f:
            lines = [i for i, l in enumerate(f) if l.startswith("##")]

    vcf = pd.read_csv(vcf_path, sep='\t', skiprows=lines)

    if filter:
        vcf = vcf.loc[vcf.FILTER == "PASS"]

    return vcf


def get_variant_id(vcf, include_alt=True):

    output = vcf["#CHROM"].astype(str) + ":" + vcf["POS"].astype(int).astype(str) + ":" + vcf["REF"].astype(str)

    if include_alt:
        return output + ":" + vcf["ALT"]

    else:
        return output


def get_variant_pos_id(vcf, include_alt=True):
    if "Location" in vcf.columns:
        return vcf.Location.astype(str)

    else:
        output = vcf["#CHROM"].astype(str) + ":" + vcf["POS"].astype(int).astype(str)

        return output


def remove_low_purity_samples(var_ids, purity_thresh=0.2):
    purities = get_sample_purities()
    samples = purities.loc[purities > purity_thresh].index.values

    return var_ids.loc[var_ids.sample_id.isin(samples)]


def get_TCGA_data(nodal_source="TCGA", weird_patient=('TCGA-HC-A9TE', ), remove_patients=False):

    norm_exp_TCGA = pd.read_csv(INPUT_PATH + "norm_exp_data_PRAD_processed.txt", sep="\t", index_col=0)
    exp_TCGA = pd.read_csv(INPUT_PATH + "exp_data_PRAD_processed.txt", sep="\t", index_col=0)

    if nodal_source.upper() == "TCGA":
        print("Using TCGA nodal labels.")
        nodal_status = pd.read_csv(INPUT_PATH + "nodal_status_PRAD_processed.txt", sep="\t", index_col=0)

    else:
        print("Using Broad nodal labels.")
        nodal_status = pd.read_csv(INPUT_PATH + "nodal_status_PRAD_broad_processed.txt", sep="\t", index_col=0)

    return exp_TCGA, norm_exp_TCGA, nodal_status


def get_TCGA_data_as_one(nodal_source="TCGA"):
    exp_TCGA, norm_exp_TCGA, nodal_status = get_TCGA_data(nodal_source=nodal_source)
    nodal_positive_samples = nodal_status.index.values[nodal_status.values.flatten() == "N1"]
    nodal_negative_samples = nodal_status.index.values[nodal_status.values.flatten() == "N0"]

    exp_TCGA.columns = [s + "_LN" if s in nodal_positive_samples else s + "_RP" if s in nodal_negative_samples else s
                        for s in exp_TCGA.columns]
    norm_exp_TCGA.columns = [s + "_NL" for s in norm_exp_TCGA.columns]
    exp_df = pd.concat([exp_TCGA, norm_exp_TCGA], axis=1)

    return exp_df


def read_in_all_signatures(return_index2biology=True, min_ngenes=None):
    signs = ("PCa_signatures_RB_NPEC.csv", "cancer_hallmark_signatures.csv",
             "immune_signatures.csv", "msig_db_hallmarks.csv")

    signatures, index2biology = [], {}

    for sign in signs:
        signature_df, index2biology_ = get_signatures(collection_name=sign, return_index2biology=True)
        print(signature_df.shape)
        index2biology.update(index2biology_)
        signatures += [signature_df]

    signatures = pd.concat(signatures, axis=0, sort=True)

    if min_ngenes is not None:
        sign_sizes, _ = get_signature_sizes(signatures=signatures)
        sign_sizes = pd.Series(sign_sizes)
        valid_signatures = sign_sizes.loc[sign_sizes >= min_ngenes]
        signatures = signatures.loc[valid_signatures.index]

    if return_index2biology:
        return signatures, index2biology

    else:
        return signatures


def remove_tissue_specific_genes(exp_df):
    exp_df = exp_df.transpose()
    tissue_dict = split_in_tissue_types(exp_df)

    union = []

    for tissue, df in tissue_dict.items():
        sigma = df.std(ddof=0)

        mask = sigma < 1e-5
        tissue_zero_genes = df.columns.values[mask.values]
        union = np.union1d(tissue_zero_genes, union)

    other_genes = np.setdiff1d(exp_df.columns.values, union)

    print("Removing %i genes that are not expressed in one of the tissue types." % len(union))

    return exp_df[other_genes].transpose()


def get_zscore(df, norm_patients=False, reference_tissue=None, ddof=1):
    """
    :param df: a sample by gene DF
    :return:
    """
    tissue_types = get_tissue_type(df)
    print(tissue_types)
    print(reference_tissue)
    if norm_patients:
        stat_df = df.copy()
        df = df.copy()

    elif reference_tissue is not None:
        mask = np.array([reference_tissue in ts for ts in tissue_types])
        print(mask)
        df = df.copy().transpose()

        assert mask.sum() > 0, "The reference tissue is not understood."
        stat_df = df.loc[mask]

    else:
        df = df.transpose()
        stat_df = df.copy()

    print("Calculating mean and standard deviation over %i samples." % stat_df.shape[0])

    mu, sigma = stat_df.mean(), stat_df.std(ddof=ddof)
    mask = sigma > 1e-2

    df = (df - mu)/sigma

    df = df.loc[:, mask]

    if not norm_patients:
        df = df.transpose()

    return df


def get_clinical_data_survival(clinfile=INPUT_PATH + "/gdac.broadinstitute.org_PRAD.Merge_Clinical.Level_1.2016012800.0.0/PRAD.clin.merged.txt",
                               os_file="/home/mlarmuse/Documents/Projects/TCGAmut/updated_clinical_information.csv"
                               ):

    clin_table = pd.read_csv(clinfile, sep="\t", index_col=0, header=None)
    clin_table.columns = [pat.upper() for pat in clin_table.loc["patient.bcr_patient_barcode"]]

    rename_dict = {"patient.age_at_initial_pathologic_diagnosis": "age",
                   "patient.stage_event.tnm_categories.pathologic_categories.pathologic_n": "LNI"}

    clin_table = clin_table.rename(index=rename_dict).transpose()[["age", "LNI"]]
    clin_table = clin_table.loc[clin_table.LNI.isin({"n0", "n1"})]

    os_info = pd.read_csv(os_file, index_col=1)[["PFI", "PFI.time", "OS", "OS.time", "DFI", "DFI.time"]]

    combined_table = pd.merge(clin_table, os_info, left_index=True, right_index=True)

    return combined_table


def get_clinical_info_UZ(file=INPUT_PATH + "Sample overview.csv"):
    gleason_scores = pd.read_csv(file, sep=",")
    gleason_scores.columns = [s.strip() for s in gleason_scores.columns]
    gleason_scores = gleason_scores.loc[gleason_scores.Gleason.astype(str) != "nan"]
    gleason_scores["Foci"] = gleason_scores["Foci"].str.replace("-", "_")
    gleason_scores = gleason_scores.set_index("Foci")

    return gleason_scores


def get_signature_sizes(exp=None, signatures=None, min_ngenes=None):

    if exp is None:
        exp = get_expression()
        exp = exp.transpose()

    if signatures is None:
        signatures, _ = read_in_all_signatures(return_index2biology=True, min_ngenes=min_ngenes)

    signature2size = {s: len(np.intersect1d(exp.columns.values, signatures.loc[s].astype(str)))
                      for s in signatures.index.values}

    uniq_sizes = set(list(signature2size.values()))
    return signature2size, uniq_sizes


def process_GS_scores():
    clin_info = get_clinical_info_UZ()
    clin_info["Gleason"] = clin_info["Gleason"].apply(lambda s: "+".join(s.split("+")[:2]))
    uniq_gs = np.unique(clin_info.Gleason)
    gs2num = {gs: i for i, gs in enumerate(uniq_gs)}
    clin_info["GS_num"] = clin_info.Gleason.apply(lambda s: gs2num[s])

    return clin_info


def get_index_lesions():

    GS_scores = process_GS_scores()
    GS_scores = GS_scores.loc[["RP" in s for s in GS_scores.index.values]]
    GS_scores_pats = [s.split("_")[1] for s in GS_scores.index.values]
    index_lesions = GS_scores[["GS_num", "ID"]].groupby(GS_scores_pats).idxmax()

    return index_lesions


def get_varseeds(variant_scores=INPUT_PATH + "RP_2_MLN_pvals_clear_depth6.csv"):

    pure_samples = get_samples_of_sufficient_purity()
    variant_scores = pd.read_csv(variant_scores, index_col=0)
    variant_scores = variant_scores[[s for s in pure_samples if "MLN" in s]]
    variant_scores = variant_scores.loc[[s in pure_samples for s in variant_scores.index.values]]

    r_patient_ids = np.array([s.split("_")[1] for s in variant_scores.index.values])
    c_patient_ids = np.array([s.split("_")[1] for s in variant_scores.columns.values])

    uniq_patients = np.unique(r_patient_ids)
    varseeds = []

    for pat in uniq_patients:
        pat_vars = variant_scores.loc[r_patient_ids == pat].loc[:, c_patient_ids == pat]

        if pat_vars.shape[1] > 0:
            most_seeding_var, seed_var = SeedsTargetsMatrix(pat_vars).get_rank_score()

            varseeds.append(seed_var)

    return varseeds


def get_confirmed_patients(signature_votes=FINAL_RESULTS_PATH + "different_seed_scoring_schemes_signatures.csv",
                           variant_scores=INPUT_PATH + "RP_2_MLN_pvals_clear_depth6.csv"):
    """
    Helper script to get all patients that are found by both somatic variants and signature voting
    :return:
    """
    seeds_signatures = get_signature_voting_seeds(signature_votes=signature_votes)
    var_seeds = get_varseeds(variant_scores=variant_scores)

    confirmed_lesions = np.intersect1d(seeds_signatures, var_seeds)
    confirmed_patients = [s.split("_")[1] for s in confirmed_lesions]

    return confirmed_patients


def get_signature_voting_seeds(signature_votes=FINAL_RESULTS_PATH + "different_seed_scoring_schemes_signatures.csv"):

    """
    Helper script to get all predicted seeds from the signature voting
    :return:
    """
    try:
        signature_scores = pd.read_csv(signature_votes, index_col=0)

    except FileNotFoundError:
        raise IOError("File %s not found!!" % signature_votes)

    patients = np.array([s.split("_")[1] for s in signature_scores.index.values])
    uniq_patients = np.unique(patients)

    seeds = [signature_scores["Votes"].loc[pat == patients].idxmax() for pat in uniq_patients]

    return seeds


def get_signature_voting_nonseeds(signature_votes=FINAL_RESULTS_PATH + "different_seed_scoring_schemes_signatures.csv"):

    """
    Helper script to get all predicted non-seeds from the signature voting
    The non-seeds are the lesion that receive the lowest number of votes in a patient
    :return:
    """
    try:
        signature_scores = pd.read_csv(signature_votes, index_col=0)

    except FileNotFoundError:
        raise IOError("File %s not found!!" % signature_votes)

    patients = np.array([s.split("_")[1] for s in signature_scores.index.values])
    uniq_patients = np.unique(patients)

    seeds = [signature_scores["Votes"].loc[pat == patients].idxmin() for pat in uniq_patients]

    return seeds
