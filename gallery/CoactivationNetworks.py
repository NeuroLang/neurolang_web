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

from gallery import data_utils, metafc

# %%
data_dir = Path("neurolang_data")


# %%
def xyz_to_ijk(x, y, z, mni_mask):
    voxels = nib.affines.apply_affine(
        np.linalg.inv(mni_mask.affine), np.c_[x, y, z],
    ).astype(int)
    return voxels


# %%
def init_frontend():
    nl = NeurolangPDL()

    nl.add_symbol(
        np.log, name="log", type_=Callable[[float], float],
    )

    return nl


# %%
def load_studies(nl, peak_reported, study_ids):
    n_studies_selected = int(len(study_ids) * 0.01)
    study_ids = study_ids.sample(n_studies_selected)
    peak_reported = peak_reported.loc[
        peak_reported.study_id.isin(study_ids.study_id)
    ]

    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )
    nl.add_tuple_set(
        np.round(
            nib.affines.apply_affine(
                mni_mask.affine, np.transpose(mni_mask.get_fdata().nonzero())
            )
        ).astype(int),
        name="Voxel",
    )
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_tuple_set(
        {("aCC", -2, 46, -4), ("CS", -34, -26, 60), ("lIPS", -26, -58, 48),},
        name="SeedVoxel",
    )


# %%
def load_regions_and_peaks_reported(nl):
    with nl.scope as e:
        e.RegionReported[e.region, e.s] = (
            e.PeakReported(e.x1, e.y1, e.z1, e.s)
            & e.SeedVoxel(e.region, e.x, e.y, e.z)
            & (e.d == e.EUCLIDEAN(e.x, e.y, e.z, e.x1, e.y1, e.z1))
            & (e.d < 10)
        )
        region_reported = nl.query(
            (e.region, e.s), e.RegionReported(e.region, e.s)
        )
    region_reported = region_reported.as_pandas_dataframe()
    nl.add_tuple_set(region_reported, name="RegionReported")

    with nl.scope as e:
        e.VoxelReported[e.x, e.y, e.z, e.s] = (
            e.PeakReported(e.x2, e.y2, e.z2, e.s)
            & e.Voxel(e.x, e.y, e.z)
            & (e.d == e.EUCLIDEAN(e.x, e.y, e.z, e.x2, e.y2, e.z2))
            & (e.d < 10)
        )
        voxel_reported = nl.query(
            (e.x, e.y, e.z, e.s), e.VoxelReported(e.x, e.y, e.z, e.s)
        )
    voxel_reported = voxel_reported.as_pandas_dataframe()
    nl.add_tuple_set(voxel_reported, name="VoxelReported")


# %%
resolution = 3
mni_mask = metafc.load_mni_atlas(data_dir, resolution=resolution)

# %%
coord_type = "xyz"
tfidf_threshold = 1e-2
_, peak_reported, study_ids = metafc.load_neuroquery(
    data_dir, mni_mask, tfidf_threshold=tfidf_threshold, coord_type=coord_type
)

# %%
nl = init_frontend()

# %%
load_studies(nl, peak_reported, study_ids)
load_regions_and_peaks_reported(nl)

# %%
query = r"""ProbActivationGivenSeedActivation(x, y, z, region, PROB(x, y, z, region)) :- (VoxelReported(x, y, z, s) & SelectedStudy(s)) // (RegionReported(region, s) & SelectedStudy(s))
ProbActivationGivenSeedDeactivation(x, y, z, region, PROB(x, y, z, region)) :- (VoxelReported(x, y, z, s) & SelectedStudy(s)) // (~RegionReported(region, s) & SelectedStudy(s) & SeedVoxel(region, x_s, y_s, z_s))
CountStudies(count(s)) :- Study(s)
ProbActivation(x, y, z, PROB(x, y, z)) :- VoxelReported(x, y, z, s) & SelectedStudy(s)
ProbActivationSeed(region, PROB(region)) :- RegionReported(region, s) & SelectedStudy(s)
ProbActivationAndSeedActivation(x, y, z, region, PROB(x, y, z, region)) :- VoxelReported(x, y, z, s) & SelectedStudy(s) & RegionReported(region, s)
Query(x,y,z,region,pA,pASeed,pAgA,pAgD,pAaA,n,m,kk,N, llr) :- ProbActivationAndSeedActivation(x, y, z, region, pAaA), ProbActivationGivenSeedActivation(x, y, z, region, pAgA), ProbActivationGivenSeedDeactivation(x, y, z, region, pAgD), ProbActivation(x, y, z, pA), ProbActivationSeed(region, pASeed), CountStudies(N), (m == pA * N), (n == pASeed * N), (kk == pAaA * N), ( llr == ( (kk * log(pAgA)) + (((n - kk) * log(1 - pAgA)) + (((m - kk) * log(pAgD)) + ((N - n - m + kk) * log(1 - pAgD))) ))

"""
# (llr == ((kk * log(pAgA) + (n - kk) * log(1 - pAgA) + (m - kk) * log(pAgD) + (N - n - m + kk) * log(1 - pAgD)) - (kk * log(pA) + (n - kk) * log(1 - pA) + (m - kk) * log(pA) + (N - n - m + kk) * log(1 - pA))))




# %%
with nl.scope:
    nl.execute_datalog_program(query)
    res = nl.solve_all()

# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, query)
qw

# %%
