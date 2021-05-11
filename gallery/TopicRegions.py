# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable, Iterable

import nibabel as nib
import nilearn.datasets as datasets
import nilearn.image as image
import numpy as np
import pandas as pd
import sklearn
from neurolang.frontend import NeurolangPDL
from scipy.stats import binom_test, kurtosis, norm, skew

from gallery import metafc

# %%
def init_frontend():
    nl = NeurolangPDL()

    nl.add_symbol(
        lambda it: float(sum(it)),
        name="agg_sum",
        type_=Callable[[Iterable], float],
    )

    @nl.add_symbol
    def agg_count(*iterables) -> int:
        return len(next(iter(iterables)))

    @nl.add_symbol
    def percentile95(p):
        return np.percentile(p, 90)

    @nl.add_symbol
    def Mean(p):
        return np.mean(p)

    @nl.add_symbol
    def STD(p):
        return np.std(p)

    @nl.add_symbol
    def Kurtosis(p):
        return kurtosis(p)

    @nl.add_symbol
    def Skew(p):
        return skew(p)

    @nl.add_symbol
    def score_test(p, p0):
        return (p - p0) / (np.sqrt(p0 * (1 - p0) / 10000))

    @nl.add_symbol
    def wald_test(p, p0, se):
        return (p - p0) / (se)

    @nl.add_symbol
    def binomial_test(p, p0):
        return norm.ppf(
            binom_test(int(p * 10000), 10000, p0, alternative="greater")
        )

    @nl.add_symbol
    def log_odds(p, p0):
        logodds = np.log((p / (1 - p)) / (p0 / (1 - p0)))
        return logodds

    return nl


def load_studies(nl, term_in_study, peak_reported, study_ids):
    PeakReported = nl.add_tuple_set(peak_reported, name="PeakReported")
    TermInStudy = nl.add_tuple_set(term_in_study, name="TermInStudy")

    study_ids = study_ids.reset_index().drop_duplicates()
    study_ids = study_ids.rename(columns={"index": "idxs"})
    Study = nl.add_tuple_set(study_ids[["idxs", "study_id"]], name="Study")
    SelectedStudy = nl.add_uniform_probabilistic_choice_over_set(
        study_ids[["study_id"]], name="SelectedStudy"
    )


def load_regions(data_dir: Path, nl, region_voxels, difumo_meta):
    Region = nl.add_tuple_set(
        {(row["Difumo_names"],) for _, row in difumo_meta.iterrows()},
        name="Region",
    )

    RegionVoxel = nl.add_tuple_set(region_voxels, name="RegionVoxel")

    coactivation_mat = pd.read_csv(
        data_dir / "topic_regions" / "coactivation_matrix.csv", index_col=0
    )
    coactivation_mat.fillna(0).shape
    grad = np.load(data_dir / "topic_regions" / "gradients.npy")
    bins = np.load(data_dir / "topic_regions" / "bins.npy")

    labels_idx = np.where(
        region_voxels["label"].values[:, None] == coactivation_mat.index.values
    )
    # FIXME: Should be :
    # lpfc_regions = region_voxels.iloc[labels_idx[0]]
    lpfc_regions = region_voxels.iloc[labels_idx[0]][: len(grad)]
    lpfc_regions.insert(4, "Gradient", grad[: len(labels_idx[0])], True)
    lpfc_regions.insert(0, "Bins", bins[: len(labels_idx[0])], True)

    bins_labels = np.load(data_dir / "topic_regions" / "bins_labels.npy")
    LpfcArea = nl.add_tuple_set(
        lpfc_regions.sort_values(by=["Bins"]), name="LpfcArea"
    )
    bins_regions = pd.DataFrame(bins_labels, columns=["bin"])
    bins_regions["Difumo_regions"] = coactivation_mat.index.values
    BinRegion = nl.add_tuple_set(
        bins_regions.sort_values(by=["bin"]), name="BinRegion"
    )
    Bin = nl.add_tuple_set(bins_labels, name="Bin")
    return lpfc_regions


def load_images(data_dir: Path, mni_mask, lpfc_regions):
    lpfc_mask = image.resample_to_img(
        source_img=nib.load(data_dir / "topic_regions" / "lfc_mask.nii.gz"),
        target_img=mni_mask,
        interpolation="nearest",
    )

    farray = np.zeros_like(mni_mask.get_fdata())
    for a, b, c, d in zip(
        lpfc_regions.i, lpfc_regions.j, lpfc_regions.k, lpfc_regions.Bins
    ):
        farray[a, b, c] = d

    farray[np.where(farray != 0)]

    mni_hires = image.resample_img(
        nib.load(datasets.fetch_icbm152_2009()["gm"]), np.eye(3) * 1
    )

    gradient_img = image.resample_to_img(
        source_img=nib.Nifti1Image(dataobj=farray, affine=mni_mask.affine),
        target_img=mni_hires,
        interpolation="nearest",
    )


def load_topics(data_dir: Path, nl):
    features = pd.read_csv(
        data_dir / "topic_regions" / "v3-topics-50.txt", sep="\t", index_col=0
    )
    topics_to_keep = [
        1,
        4,
        6,
        14,
        18,
        19,
        23,
        25,
        20,
        21,
        27,
        29,
        30,
        31,
        33,
        35,
        36,
        38,
        37,
        41,
        44,
        45,
        48,
        49,
    ]

    labels = [
        "face/affective processing",
        "verbal semantics",
        "cued attention",
        "working memory",
        "autobiographical memory",
        "reading",
        "inhibition",
        "motor",
        "visual perception",
        "numerical cognition",
        "reward-based decision making",
        "visual attention",
        "multisensory processing",
        "visuospatial",
        "eye movements",
        "action",
        "auditory processing",
        "pain",
        "language",
        "declarative memory",
        "visual semantics",
        "emotion",
        "cognitive control",
        "social cognition",
    ]

    features = features.iloc[:, topics_to_keep]
    features.columns = labels
    topix = pd.DataFrame(features.unstack().reset_index())
    topix = topix.rename(columns={0: "Prob"})
    topix = topix.rename(columns={"level_0": "Topic"})

    TopicInStudy = nl.add_probabilistic_facts_from_tuples(
        topix[["Prob", "Topic", "id"]].itertuples(index=False),
        name="TopicInStudy",
    )


def load_study_splits(
    nl, study_ids, n_splits: int = 1000, split_proportion: float = 0.6
):
    splits = []
    for i, (train, _) in enumerate(
        sklearn.model_selection.ShuffleSplit(
            n_splits=n_splits, train_size=split_proportion,
        ).split(study_ids)
    ):
        split = study_ids.iloc[train].copy().rename(columns={0: "study_id"})
        split["split_id"] = i
        splits.append(split)
    splits = pd.concat(splits)
    StudySplit = nl.add_tuple_set(splits, name="StudySplit")


# %%
data_dir = Path("neurolang_data")

# %%
resolution = 2
interpolation = "nearest"
mni_mask = metafc.load_mni_atlas(
    data_dir, resolution=resolution, interpolation=interpolation
)

# %%
coord_type = "ijk"
tfidf_threshold = 0.01
term_in_study, peak_reported, study_ids = metafc.load_neuroquery(
    data_dir, mni_mask, tfidf_threshold=tfidf_threshold, coord_type=coord_type
)

# %%
n_difumo_components = 1024
region_voxels, difumo_meta = metafc.load_difumo(
    data_dir,
    mni_mask,
    n_difumo_components=n_difumo_components,
    coord_type=coord_type,
)

# %%
nl = init_frontend()

# %%
load_studies(nl, term_in_study, peak_reported, study_ids)
load_regions(data_dir, nl, region_voxels, difumo_meta)


# %%
load_topics(data_dir, nl)
load_study_splits(nl, study_ids, n_splits=10)

# %%
query = r"""BinReported(b, s) :- PeakReported(i, j, k, s) & LpfcArea(b, r, i, j, k, g)
StudyNotReportingPeak(b, s) :- Study(idxs, s) & Bin(b) & ~BinReported(b, s)
BinAux(b, s) :- Bin(b2) & (b2 != b) & BinReported(b2, s)
StudyMatchingBinSegregationQuery(s, b) :- BinReported(b, s) & ~BinAux(b, s)
ProbTopicInStudyA(t, split, PROB(t, split)) :- TopicInStudy(t, s) & SelectedStudy(s) & StudySplit(idxs, s, split)
ProbTopicStudy(t, Mean(p)) :- ProbTopicInStudyA(t, split, p)
QueryA(t, b, split, PROB(t, b, split)) :- TopicInStudy(t, s) // (StudyMatchingBinSegregationQuery(s, b) & SelectedStudy(s) & StudySplit(idxs, s, split))
QueryB(t, b, split, PROB(t, b, split)) :- (TopicInStudy(t, s)) // (StudyNotReportingPeak(b, s) & SelectedStudy(s) & StudySplit(idxs, s, split))
QueryP(t, b, Mean(p0)) :- QueryA(t, b, split, p0)
QueryN(t, b, Mean(p1)) :- QueryB(t, b, split, p1)
ThrQuery(t, b, p, pmarg, zw) :- QueryP(t, b, p) & QueryN(t, b, pmarg) & (zw == log_odds(p, pmarg))"""

# %% tags=[]
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, query)
qw

# %%
