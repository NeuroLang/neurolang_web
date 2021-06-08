import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import nibabel
import nilearn.datasets
import nilearn.datasets.utils
import nilearn.image
import numpy as np
import pandas as pd
import scipy.sparse
from neurolang.frontend.neurosynth_utils import StudyID


DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID = {
    128: "wjvd5",
    256: "3vrct",
    512: "9b76y",
    1024: "34792",
}


def xyz_to_ijk(xyz, mask):
    voxels = nibabel.affines.apply_affine(
        np.linalg.inv(mask.affine),
        xyz,
    ).astype(int)
    return voxels


def read_and_convert_csv_to_feather(file: Union[Path, str], **csv_read_args):
    """
    Load a target csv file. If this is the first time this file is read,
    this method will save the file to feather file format. Later calls to
    this method to read the csv file will then read the file from the
    .feather file instead of from the .csv file, speeding up read times.

    Parameters
    ----------
    file : Union[Path, str]
        the path of the csv file
    **csv_read_args :
        additional args to pass to pandas read_csv method

    Returns
    -------
    pd.DataFrame
        the loaded dataframe
    """
    if not isinstance(file, Path):
        file = Path(file)
    target_file = file.with_suffix(".feather")
    if not os.path.isfile(target_file):
        df = pd.read_csv(file, **csv_read_args)
        df.to_feather(target_file)
    else:
        df = pd.read_feather(target_file)
    return df


def read_and_convert_csv_to_hdf(file: Union[Path, str], **csv_read_args):
    """
    Load a target csv file. If this is the first time this file is read,
    this method will save the file to hdf fixed file format. Later calls to
    this method to read the csv file will then read the file from the
    hdf file instead of from the .csv file, speeding up read times.

    Parameters
    ----------
    file : Union[Path, str]
        the path of the csv file
    **csv_read_args :
        additional args to pass to pandas read_csv method

    Returns
    -------
    pd.DataFrame
        the loaded dataframe
    """
    if not isinstance(file, Path):
        file = Path(file)
    target_file = file.with_suffix(".h5")
    if not os.path.isfile(target_file):
        df = pd.read_csv(file, **csv_read_args)
        df.to_hdf(target_file, "data", mode="w")
    else:
        df = pd.read_hdf(target_file, "data")
    return df


def fetch_neuroquery(
    mask: nibabel.Nifti1Image,
    data_dir: Path,
    tfidf_threshold: Optional[float] = None,
    coord_type: str = "xyz",
    convert_study_ids: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    peak_data, study_ids = fetch_neuroquery_peak_data(
        mask,
        data_dir=data_dir,
        coord_type=coord_type,
        convert_study_ids=convert_study_ids,
    )
    term_data = fetch_neuroquery_term_data(
        data_dir=data_dir,
        tfidf_threshold=tfidf_threshold,
        convert_study_ids=convert_study_ids,
    )
    return term_data, peak_data, study_ids


def fetch_neuroquery_term_data(
    data_dir: Path,
    tfidf_threshold: Optional[float] = None,
    convert_study_ids: bool = True,
) -> pd.DataFrame:
    """
    Fetch neuroquery term_data
    """
    base_url = "https://github.com/neuroquery/neuroquery_data/"
    tfidf_url = base_url + "raw/master/training_data/corpus_tfidf.npz"
    feature_names_url = base_url + "raw/master/training_data/feature_names.txt"
    study_ids_url = base_url + "raw/master/training_data/pmids.txt"
    out_dir = data_dir / "neuroquery"
    (tfidf_fn, feature_names_fn, study_ids_fn,) = nilearn.datasets.utils._fetch_files(
        out_dir,
        [
            ("corpus_tfidf.npz", tfidf_url, dict()),
            ("feature_names.txt", feature_names_url, dict()),
            ("pmids.txt", study_ids_url, dict()),
        ],
    )
    tfidf = scipy.sparse.load_npz(tfidf_fn)
    feature_names = read_and_convert_csv_to_hdf(feature_names_fn, header=None)
    study_ids = read_and_convert_csv_to_hdf(study_ids_fn, header=None)
    study_ids.rename(columns={0: "study_id"}, inplace=True)
    if convert_study_ids:
        study_ids["study_id"] = study_ids["study_id"].apply(StudyID)
    tfidf = pd.DataFrame(tfidf.todense(), columns=feature_names[0])
    tfidf["study_id"] = study_ids.iloc[:, 0]
    return create_and_save_term_data(out_dir, tfidf, tfidf_threshold=tfidf_threshold)


def fetch_neuroquery_peak_data(
    mask: nibabel.Nifti1Image,
    data_dir: Path,
    coord_type: str = "xyz",
    convert_study_ids: bool = True,
):
    """
    Fetch peak_data and study_ids from neuroquery atlas.
    """
    base_url = "https://github.com/neuroquery/neuroquery_data/"
    coordinates_url = base_url + "raw/master/training_data/coordinates.csv"
    study_ids_url = base_url + "raw/master/training_data/pmids.txt"
    out_dir = data_dir / "neuroquery"
    (coordinates_fn, study_ids_fn,) = nilearn.datasets.utils._fetch_files(
        out_dir,
        [
            ("coordinates.csv", coordinates_url, dict()),
            ("pmids.txt", study_ids_url, dict()),
        ],
    )
    coordinates = read_and_convert_csv_to_hdf(coordinates_fn)
    assert coord_type in ("xyz", "ijk")
    if coord_type == "ijk":
        ijk = xyz_to_ijk(coordinates[["x", "y", "z"]], mask)
        coordinates["i"] = ijk[:, 0]
        coordinates["j"] = ijk[:, 1]
        coordinates["k"] = ijk[:, 2]
    coord_cols = list(coord_type)
    peak_data = coordinates[coord_cols + ["pmid"]].rename(columns={"pmid": "study_id"})
    study_ids = read_and_convert_csv_to_hdf(study_ids_fn, header=None)
    study_ids.rename(columns={0: "study_id"}, inplace=True)
    if convert_study_ids:
        peak_data["study_id"] = peak_data["study_id"].apply(StudyID)
        study_ids["study_id"] = study_ids["study_id"].apply(StudyID)

    return peak_data, study_ids


def fetch_neurosynth_peak_data(
    data_dir: Path, convert_study_ids: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch peak_data and study_ids from neurosynth atlas.
    """
    ns_dir = data_dir / "neurosynth"
    ns_data_url = "https://github.com/neurosynth/neurosynth-data/raw/master/"
    ns_database_fn = nilearn.datasets.utils._fetch_files(
        ns_dir,
        [
            (
                "database.txt",
                ns_data_url + "current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )[0]
    activations = read_and_convert_csv_to_hdf(ns_database_fn, sep="\t")
    if convert_study_ids:
        activations["id"] = activations["id"].apply(StudyID)
    mni_peaks = activations.loc[activations.space == "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})
    non_mni_peaks = activations.loc[activations.space != "MNI"][
        ["x", "y", "z", "id"]
    ].rename(columns={"id": "study_id"})
    proj_mat = np.linalg.pinv(
        np.array(
            [
                [0.9254, 0.0024, -0.0118, -1.0207],
                [-0.0048, 0.9316, -0.0871, -1.7667],
                [0.0152, 0.0883, 0.8924, 4.0926],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ).T
    )
    projected = np.round(
        np.dot(
            np.hstack(
                (
                    non_mni_peaks[["x", "y", "z"]].values,
                    np.ones((len(non_mni_peaks), 1)),
                )
            ),
            proj_mat,
        )[:, 0:3]
    )
    projected_df = pd.DataFrame(
        np.hstack([projected, non_mni_peaks[["study_id"]].values]),
        columns=["x", "y", "z", "study_id"],
    )
    peak_data = pd.concat([projected_df, mni_peaks]).astype(
        {"x": int, "y": int, "z": int}
    )
    study_ids = peak_data[["study_id"]].drop_duplicates()
    return peak_data, study_ids


def fetch_neurosynth_term_data(
    data_dir: Path,
    tfidf_threshold: Optional[float] = None,
    convert_study_ids: bool = True,
) -> pd.DataFrame:
    """
    Fetch neurosynth term_data
    """
    ns_dir = data_dir / "neurosynth"
    ns_data_url = "https://github.com/neurosynth/neurosynth-data/raw/master/"
    ns_features_fn = nilearn.datasets.utils._fetch_files(
        ns_dir,
        [
            (
                "features.txt",
                ns_data_url + "current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )[0]
    features = read_and_convert_csv_to_hdf(ns_features_fn, sep="\t")
    features.rename(columns={"pmid": "study_id"}, inplace=True)

    if convert_study_ids:
        features["study_id"] = features["study_id"].apply(StudyID)
    return create_and_save_term_data(ns_dir, features, tfidf_threshold=tfidf_threshold)


def fetch_neurosynth(
    data_dir: Path,
    tfidf_threshold: Optional[float] = None,
    convert_study_ids: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    peak_data, study_ids = fetch_neurosynth_peak_data(
        data_dir=data_dir, convert_study_ids=convert_study_ids
    )
    term_data = fetch_neurosynth_term_data(
        data_dir=data_dir,
        tfidf_threshold=tfidf_threshold,
        convert_study_ids=convert_study_ids,
    )
    return term_data, peak_data, study_ids


def create_and_save_term_data(
    data_dir: Path,
    df,
    var_name: str = "term",
    id_vars: str = "study_id",
    value_name: str = "tfidf",
    tfidf_threshold: Optional[float] = None,
):
    """
    This method creates the term_data dataframe by changing the shape of the given df using pandas' melt function.
    Because this melt operation is a bit slow (~5s.), in order to speedup data loading time, we save the resulting
    dataframe on file so that it can be read directly next time instead of calling the melt function.

    Parameters
    ----------
    data_dir : Path
        the data directory where the resulting file will be saved if it doesnt already exist
    df : pd.Dataframe
        the input dataframe, with n rows and p columns, n being the study ids and p the terms. The values of the input
        dataframe are the tfidf values for study n and term p
    var_name : str, optional
        the name for the variable column, by default "term"
    id_vars : str, optional
        the name of the id column, by default "study_id"
    value_name : str, optional
        the name of the output value column, by default "tfidf"
    tfidf_threshold : Optional[float], optional
        the minimum threshold for which a tfidf value is kept, by default None

    Returns
    -------
    pd.Dataframe
        a Dataframe of x rows and 3 columns (study_id, term, tfidf)
    """
    target_file = data_dir / f"tfidf_{str(tfidf_threshold).replace('.', '_')}.h5"
    if os.path.isfile(target_file):
        term_data = pd.read_hdf(target_file, "data")
    else:
        term_data = pd.melt(
            df,
            var_name=var_name,
            id_vars=id_vars,
            value_name=value_name,
        )
        if tfidf_threshold is not None:
            term_data = term_data.query(f"tfidf > {tfidf_threshold}")[
                [var_name, value_name, id_vars]
            ]
        else:
            term_data = term_data.query("tfidf > 0")[[var_name, value_name, id_vars]]
        term_data.to_hdf(target_file, "data", mode="w")
    return term_data


def fetch_difumo(
    mask: nibabel.Nifti1Image,
    data_dir: Path,
    component_filter_fun: Callable = lambda _: True,
    coord_type: str = "xyz",
    n_components: int = 256,
) -> Tuple[pd.DataFrame, nibabel.Nifti1Image]:
    out_dir = data_dir / "difumo"
    download_id = DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID[n_components]
    url = f"https://osf.io/{download_id}/download"
    csv_file = os.path.join(str(n_components), f"labels_{n_components}_dictionary.csv")
    nifti_file = os.path.join(str(n_components), "3mm/maps.nii.gz")
    files = [
        (csv_file, url, {"uncompress": True}),
        (nifti_file, url, {"uncompress": True}),
    ]
    files = nilearn.datasets.utils._fetch_files(out_dir, files, verbose=2)
    labels = read_and_convert_csv_to_hdf(files[0])
    img = nilearn.image.load_img(files[1])
    img = nilearn.image.resample_img(
        img,
        target_affine=mask.affine,
        interpolation="nearest",
    )
    img_data = img.get_fdata()
    to_concat = list()
    for i, label in enumerate(
        labels.loc[labels.apply(component_filter_fun, axis=1)].Difumo_names
    ):
        coordinates = np.argwhere(img_data[:, :, :, i] > 0)
        if coord_type == "xyz":
            coordinates = nibabel.affines.apply_affine(img.affine, coordinates)
        else:
            assert coord_type == "ijk"
        region_data = pd.DataFrame(coordinates, columns=list(coord_type))
        region_data["label"] = label
        to_concat.append(region_data[["label"] + list(coord_type)])
    region_voxels = pd.concat(to_concat)
    return region_voxels, labels


def fetch_neurosynth_topic_associations(
    data_dir: Path,
    n_topics: int,
    convert_study_ids: bool = True,
    topics_to_keep: List[int] = None,
    labels: List[str] = None,
    version: str = "v5",
) -> pd.DataFrame:
    if n_topics not in {50, 100, 200, 400}:
        raise ValueError(f"Unexpected number of topics: {n_topics}")
    ns_dir = data_dir / "neurosynth"
    ns_data_url = "https://github.com/neurosynth/neurosynth-data/raw/master/"
    topic_data = nilearn.datasets.utils._fetch_files(
        ns_dir,
        [
            (
                f"analyses/{version}-topics-{n_topics}.txt",
                ns_data_url + f"topics/{version}-topics.tar.gz",
                {"uncompress": True},
            ),
        ],
    )[0]
    ta = read_and_convert_csv_to_hdf(topic_data, sep="\t")
    if convert_study_ids:
        ta["id"] = ta["id"].apply(StudyID)
    ta.set_index("id", inplace=True)
    if topics_to_keep is not None:
        ta = ta.iloc[:, topics_to_keep]
    if labels is not None:
        ta.columns = labels
    ta = ta.unstack().reset_index()
    ta.columns = ("topic", "study_id", "prob")
    ta = ta[["prob", "topic", "study_id"]]
    return ta


def load_mni_atlas(
    data_dir: Path,
    resolution: int = 2,
    interpolation: str = "continuous",
    key: str = "gm",
):
    """Load the MNI atlas and resample it to 2mm voxels."""

    mni_mask = nilearn.image.resample_img(
        nibabel.load(nilearn.datasets.fetch_icbm152_2009(data_dir=str(data_dir))[key]),
        np.eye(3) * resolution,
        interpolation=interpolation,
    )
    return mni_mask
