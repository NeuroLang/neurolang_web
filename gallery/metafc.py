import nibabel as nib
import nilearn.datasets as datasets
import nilearn.image as image
import numpy as np

from pathlib import Path
from gallery import data_utils


def load_mni_atlas(resolution: int = 2, interpolation: str = "continuous"):
    """Load the MNI atlas and resample it to 2mm voxels."""

    mni_mask = image.resample_img(
        nib.load(datasets.fetch_icbm152_2009()["gm"]),
        np.eye(3) * resolution,
        interpolation=interpolation,
    )
    return mni_mask


def load_neuroquery(
    data_dir: Path,
    mni_mask,
    tfidf_threshold: float = 0.01,
    coord_type: str = "xyz",
):
    term_in_study, peak_reported, study_ids = data_utils.fetch_neuroquery(
        data_dir=data_dir,
        tfidf_threshold=tfidf_threshold,
        mask=mni_mask,
        coord_type=coord_type,
    )
    return term_in_study, peak_reported, study_ids


def load_difumo(
    data_dir: Path,
    mni_mask,
    n_difumo_components: int = 1024,
    coord_type: str = "xyz",
):
    region_voxels, difumo_meta = data_utils.fetch_difumo(
        data_dir=data_dir,
        mask=mni_mask,
        coord_type=coord_type,
        n_components=n_difumo_components,
    )
    return region_voxels, difumo_meta
