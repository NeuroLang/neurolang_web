import datetime
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

DATA_DIR = Path(os.path.dirname(os.path.realpath(__file__))) / "neurolang_data"

FILETYPE_TO_EXTENSION = {
    "hdf": "h5",
    "parquet": "gz",
}

DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID = {
    128: "wjvd5",
    256: "3vrct",
    512: "9b76y",
    1024: "34792",
}


def xyz_to_ijk(xyz, mask):
    voxels = nibabel.affines.apply_affine(np.linalg.inv(mask.affine), xyz,).astype(int)
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
    data_dir: Path = DATA_DIR,
    tfidf_threshold: Optional[float] = None,
    coord_type: str = "xyz",
    convert_study_ids: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base_url = "https://github.com/neuroquery/neuroquery_data/"
    tfidf_url = base_url + "raw/master/training_data/corpus_tfidf.npz"
    coordinates_url = base_url + "raw/master/training_data/coordinates.csv"
    feature_names_url = base_url + "raw/master/training_data/feature_names.txt"
    study_ids_url = base_url + "raw/master/training_data/pmids.txt"
    out_dir = data_dir / "neuroquery"
    (
        tfidf_fn,
        coordinates_fn,
        feature_names_fn,
        study_ids_fn,
    ) = nilearn.datasets.utils._fetch_files(
        out_dir,
        [
            ("corpus_tfidf.npz", tfidf_url, dict()),
            ("coordinates.csv", coordinates_url, dict()),
            ("feature_names.txt", feature_names_url, dict()),
            ("pmids.txt", study_ids_url, dict()),
        ],
    )
    tfidf = scipy.sparse.load_npz(tfidf_fn)
    coordinates = read_and_convert_csv_to_hdf(coordinates_fn)
    assert coord_type in ("xyz", "ijk")
    if coord_type == "ijk":
        ijk = xyz_to_ijk(coordinates[["x", "y", "z"]], mask)
        coordinates["i"] = ijk[:, 0]
        coordinates["j"] = ijk[:, 1]
        coordinates["k"] = ijk[:, 2]
    coord_cols = list(coord_type)
    peak_data = coordinates[coord_cols + ["pmid"]].rename(columns={"pmid": "study_id"})
    feature_names = read_and_convert_csv_to_hdf(feature_names_fn, header=None)
    study_ids = read_and_convert_csv_to_hdf(study_ids_fn, header=None)
    study_ids.rename(columns={0: "study_id"}, inplace=True)
    if convert_study_ids:
        peak_data["study_id"] = peak_data["study_id"].apply(StudyID)
        study_ids["study_id"] = study_ids["study_id"].apply(StudyID)
    tfidf = pd.DataFrame(tfidf.todense(), columns=feature_names[0])
    tfidf["study_id"] = study_ids.iloc[:, 0]
    if tfidf_threshold is None:
        term_data = pd.melt(
            tfidf, var_name="term", id_vars="study_id", value_name="tfidf",
        ).query("tfidf > 0")[["term", "tfidf", "study_id"]]
    else:
        term_data = pd.melt(
            tfidf, var_name="term", id_vars="study_id", value_name="tfidf",
        ).query(f"tfidf > {tfidf_threshold}")[["term", "study_id"]]
    return term_data, peak_data, study_ids


def fetch_neurosynth(
    tfidf_threshold: Optional[float] = None,
    data_dir: Path = DATA_DIR,
    convert_study_ids: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ns_dir = data_dir / "neurosynth"
    ns_data_url = "https://github.com/neurosynth/neurosynth-data/raw/master/"
    ns_database_fn, ns_features_fn = nilearn.datasets.utils._fetch_files(
        ns_dir,
        [
            (
                "database.txt",
                ns_data_url + "current_data.tar.gz",
                {"uncompress": True},
            ),
            (
                "features.txt",
                ns_data_url + "current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
    )
    converters = None
    if convert_study_ids:
        converters = {"pmid": StudyID}
    features = pd.read_csv(ns_features_fn, sep="\t", converters=converters)
    features.rename(columns={"pmid": "study_id"}, inplace=True)
    term_data = pd.melt(
        features, var_name="term", id_vars="study_id", value_name="tfidf",
    )
    if tfidf_threshold is not None:
        term_data = term_data.query("tfidf > {}".format(tfidf_threshold))[
            ["term", "study_id"]
        ]
    else:
        term_data = term_data.query("tfidf > 0")[["term", "tfidf", "study_id"]]
    if convert_study_ids:
        converters = {"id": StudyID}
    activations = pd.read_csv(ns_database_fn, sep="\t", converters=converters)
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
    return term_data, peak_data, study_ids


def fetch_difumo_meta(
    data_dir: Path = DATA_DIR, n_components: int = 256,
) -> pd.DataFrame:
    out_dir = data_dir / "difumo"
    download_id = DIFUMO_N_COMPONENTS_TO_DOWNLOAD_ID[n_components]
    url = f"https://osf.io/{download_id}/download"
    labels_path = os.path.join(
        str(n_components), f"labels_{n_components}_dictionary.csv"
    )
    files = [
        (labels_path, url, {"uncompress": True}),
    ]
    files = nilearn.datasets.utils._fetch_files(out_dir, files, verbose=2)
    labels = pd.DataFrame(pd.read_csv(files[0]))
    return labels


def fetch_difumo(
    mask: nibabel.Nifti1Image,
    component_filter_fun: Callable = lambda _: True,
    data_dir: Path = DATA_DIR,
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
    labels = pd.DataFrame(pd.read_csv(files[0]))
    img = nilearn.image.load_img(files[1])
    img = nilearn.image.resample_img(
        img, target_affine=mask.affine, interpolation="nearest",
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


def save_file(d: pd.DataFrame, dst_path: Path, file_type: str = "hdf") -> None:
    print(f"saving to {dst_path}")
    if file_type == "hdf":
        with pd.HDFStore(
            dst_path, mode="w", complib="blosc:lz4", complevel=9
        ) as hdf_store:
            hdf_store["data"] = d
    else:
        d.to_parquet(dst_path)


def save_to_parquet(d: pd.DataFrame, dst_path: Path) -> None:
    print(f"saving to {dst_path}")
    d.to_parquet(dst_path, compression="gzip")


def get_exp_dir(exp_name: str):
    module_dir = Path(__file__).parent
    exp_dir = module_dir / exp_name
    if not exp_dir.is_dir():
        raise FileNotFoundError(f"Unknown exp: exp dir {exp_dir} does not exist")
    return exp_dir


def load_results(
    exp_name: str,
    out_path: Optional[Union[str, Path]] = None,
    use_cache: bool = True,
    take_last: bool = False,
    file_type: str = "hdf",
) -> pd.DataFrame:
    extension = FILETYPE_TO_EXTENSION[file_type]
    exp_dir = get_exp_dir(exp_name)
    result_dir = exp_dir / "_results"
    if not result_dir.is_dir():
        raise FileNotFoundError(
            f"Results not available for exp {exp_name}. "
            f"Directory {exp_dir} does not exist"
        )
    if use_cache:
        cache_paths = list(result_dir.glob(f"{exp_name}-results*.{extension}"))
        if cache_paths:
            path = result_dir / next(reversed(sorted(cache_paths)))
            if file_type == "hdf":
                return pd.read_hdf(path, "data")
            elif file_type == "parquet":
                return pd.read_parquet(path)
    if out_path is not None:
        if isinstance(out_path, str):
            out_path = Path(out_path)
    paths = list(
        sorted(
            p
            for p in result_dir.glob(f"*.{extension}")
            if not str(p).startswith("exp_name")
        )
    )
    if take_last:
        paths = paths[-1:]
    to_concat = list()
    for path in paths:
        if file_type == "hdf":
            to_concat.append(pd.read_hdf(result_dir / path, "data"))
        else:
            to_concat.append(pd.read_parquet(result_dir / path))
    results = pd.concat(to_concat)
    datestr = datetime.date.today().isoformat()
    cache_path = result_dir / f"{exp_name}-results-{datestr}.{extension}"
    if cache_path.is_file():
        cache_path.unlink()
    save_file(results, cache_path, file_type=file_type)
    if out_path is not None:
        save_file(results, out_path, file_type=file_type)
    return results


def load_aggregated_results(exp_name: str, file_type: str = "hdf") -> pd.DataFrame:
    extension = FILETYPE_TO_EXTENSION[file_type]
    exp_dir = get_exp_dir(exp_name)
    result_dir = exp_dir / "_results"
    if not result_dir.is_dir():
        raise FileNotFoundError(
            f"Results not available for exp {exp_name}. "
            f"Directory {exp_dir} does not exist"
        )
    cache_paths = list(result_dir.glob(f"{exp_name}-aggregated-results*.{extension}"))
    if not cache_paths:
        raise FileNotFoundError(f"Aggregated results not available in {result_dir}")
    print("loading cached aggregated results from")
    print(next(iter(cache_paths)))
    path = result_dir / next(iter(cache_paths))
    if file_type == "hdf":
        return pd.read_hdf(path, "data")
    return pd.read_parquet(path)


def save_aggregated_results(
    exp_name: str, results: pd.DataFrame, file_type: str = "hdf"
) -> None:
    extension = FILETYPE_TO_EXTENSION[file_type]
    exp_dir = get_exp_dir(exp_name)
    result_dir = exp_dir / "_results"
    if not result_dir.is_dir():
        raise FileNotFoundError(f"Directory {exp_dir} does not exist")
    datestr = datetime.date.today().isoformat()
    filename = f"{exp_name}-aggregated-results-{datestr}.{extension}"
    save_file(results, result_dir / filename, file_type=file_type)
    print("saved aggregated results to")
    print(result_dir / filename)


def load_cognitive_terms(filename: str) -> pd.Series:
    if filename is None:
        path = Path(__file__).parent / "cognitive_terms.txt"
    else:
        path = Path(__file__).parent / f"{filename}.txt"
    return pd.read_csv(path, header=None, names=["term"]).drop_duplicates()


def load_term_cogfun() -> pd.DataFrame:
    path = Path(__file__).parent / "term_cogfun.csv"
    return pd.read_csv(path)


def subsample_cbma_data(
    term_data: pd.DataFrame,
    peak_data: pd.DataFrame,
    study_ids: pd.DataFrame,
    proportion: float = None,
    nb_samples: int = None,
):
    if proportion is not None:
        assert nb_samples is None
        nb_samples = int(len(study_ids) * proportion)
    study_ids = study_ids.sample(nb_samples)
    term_data = term_data.loc[term_data.study_id.isin(study_ids.study_id)]
    peak_data = peak_data.loc[peak_data.study_id.isin(study_ids.study_id)]
    return term_data, peak_data, study_ids


def fetch_neurosynth_topic_associations(
    n_topics: int,
    data_dir: Path = DATA_DIR,
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
    converters = None
    if convert_study_ids:
        converters = {"id": StudyID}
    ta = pd.read_csv(topic_data, sep="\t", converters=converters)
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
    data_dir: Path = DATA_DIR,
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
