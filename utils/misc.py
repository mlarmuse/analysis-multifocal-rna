import numpy as np


def list_of_tuples_to_dict(l):
    keys = {t[0] for t in l}

    odict = {k: [] for k in keys}

    for t in l:
        odict[t[0]] += [t[1]]

    return odict


def get_digit(s):
    return "".join([c for c in s if c.isdigit()])


def sort_ids_by_patient_sample_type_number(ids):
    patient_numbers = np.array([int(s.split("_")[1][2:]) for s in ids])
    sample_type = np.array(["RP" if "RP" in s else "NL" if "NL" in s else "LN" for s in ids])
    sample_nr = np.array([0 if "NL" in s else int(get_digit(s.split("_")[2])) for s in ids])

    sortidxs = np.lexsort((sample_nr, sample_type, patient_numbers))

    return np.asarray(ids)[sortidxs]


def sort_patients(score_frame):
    sort_idxs = sort_ids_by_patient_sample_type_number(score_frame.index.values)

    return score_frame.loc[sort_idxs]


def get_tissue_type(sample_ids):
    return np.array([get_tissue_type_from_id(s) for s in sample_ids])


def get_tissue_type_from_id(s):
    sample_type = s.split("_")[-1]
    return ''.join([i for i in sample_type if not i.isdigit()])


def split_in_tissue_types(df):
    tissues = get_tissue_type(df.index.values)
    tissue_types = np.unique(tissues)
    odict = {type_: df.loc[tissues == type_] for type_ in tissue_types}

    return odict


def get_all_patient_samples(df, patient_id):
    mask = [s.split("_")[1] == patient_id for s in df.index.values]
    pat_df = df.loc[mask]

    return pat_df


def process_table(df):
    df = df.copy(deep=True)
    types = [s.name for s in df.dtypes.values]

    for col, type_ in zip(list(df.columns), types):
        vals = df[col].values

        if "float" in type_:
            if (np.max(np.abs(vals)) >= 100) or (np.min(np.abs(vals)) <= 0.01):
                df[col] = ["%.2E" % s for s in vals]

            else:
                df[col] = ["%.2f" % s for s in vals]

        if "int" in type_:
            if (np.max(vals) >= 100) or (np.min(vals) <= -100):
                df[col] = ["%.2E" % float(s) for s in vals]

    return df