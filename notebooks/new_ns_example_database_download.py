# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.2
#   kernelspec:
#     display_name: Python 3.8.2 64-bit ('base')
#     language: python
#     name: python38264bitbase1b15e812226d42068d829dc2c51d0e05
# ---

import warnings  # type: ignore

from nilearn import datasets, image, plotting
import pandas as pd
from neurolang import frontend as fe
from neurolang.frontend import probabilistic_frontend as pfe
from typing import Iterable
import nibabel as nib
import numpy as np
from time import time
from nlweb.viewers.query import QueryWidget

warnings.filterwarnings("ignore")

# ## Data preparation

# +
###############################################################################
# Load the MNI atlas and resample it to 4mm voxels

mni_t1 = nib.load(datasets.fetch_icbm152_2009()['t1'])
mni_t1_4mm = image.resample_img(mni_t1, np.eye(3) * 1)

# +
###############################################################################
# Load the NeuroSynth database

ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
    'neurolang',
    [
        (
            'database.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
        (
            'features.txt',
            'https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz',
            {'uncompress': True}
        ),
    ]
)

ns_database = pd.read_csv(ns_database_fn, sep=f'\t')
ijk_positions = (
    np.round(nib.affines.apply_affine(
        np.linalg.inv(mni_t1_4mm.affine),
        ns_database[['x', 'y', 'z']].values.astype(float)
    )).astype(int)
)
ns_database['i'] = ijk_positions[:, 0]
ns_database['j'] = ijk_positions[:, 1]
ns_database['k'] = ijk_positions[:, 2]

ns_features = pd.read_csv(ns_features_fn, sep=f'\t')
ns_docs = ns_features[['pmid']].drop_duplicates()
ns_terms = (
    pd.melt(
            ns_features,
            var_name='term', id_vars='pmid', value_name='TfIdf'
       )
    .query('TfIdf > 1e-3')[['pmid', 'term', 'TfIdf']]
)
ns_terms['TfIdf'] = ns_terms['TfIdf'].apply(fe.neurosynth_utils.TfIDf)
ns_terms['pmid'] = ns_terms['pmid'].apply(fe.neurosynth_utils.StudyID)
ns_database = (
    ns_database
    .query("space == 'MNI'")
    [["x", "y", "z", "i", "j", "k", "id"]]
    .rename(columns={'id': 'pmid'})
)
ns_database['pmid'] = ns_database['pmid'].apply(fe.neurosynth_utils.StudyID)

# +
###############################################################################
# Probabilistic Logic Programming in NeuroLang
# --------------------------------------------

#nl = pfe.ProbabilisticFrontend()
nl = fe.NeurolangDL()

###############################################################################
# Adding new aggregation function to build a region
@nl.add_symbol
def agg_count(
    i: Iterable
) -> np.int64:
    return np.int64(len(i))

###############################################################################
# Adding new aggregation function to build a region
@nl.add_symbol
def agg_create_region(
    i: Iterable, j: Iterable, k: Iterable
) -> fe.ExplicitVBR:
    voxels = np.c_[i, j, k]
    return fe.ExplicitVBR(
        voxels, mni_t1_4mm.affine,
        image_dim=mni_t1_4mm.shape
    )

###############################################################################
# Adding new aggregation function to build a region overlay
@nl.add_symbol
def agg_create_region_overlay(
    i: Iterable, j: Iterable, k: Iterable, p: Iterable
) -> fe.ExplicitVBR:
    voxels = np.c_[i, j, k]
    return fe.ExplicitVBROverlay(
        voxels, mni_t1_4mm.affine, p,
        image_dim=mni_t1_4mm.shape
    )


@nl.add_symbol
def agg_percentile(x: Iterable, q: float) -> float:
    ret = np.percentile(x, 95)
    print("THR", ret)
    return ret


###############################################################################
# Loading the database

activations = nl.add_tuple_set(ns_database.values, name='activations')
terms = nl.add_tuple_set(ns_terms.values, name='terms')
#docs = nl.add_uniform_probabilistic_choice_over_set(
#        ns_docs.values, name='docs'
#)
# -

# # Run interface

# +
query = r'''
voxel_terms_doc_counts(term, i, j, k, agg_count(study_id)) :- terms(study_id, term, tfidf),activations(..., ..., ..., i, j, k, study_id),term == 'fear'
voxel_terms_doc_counts(term, i, j, k, agg_count(study_id)) :- terms(study_id, term, tfidf),activations(..., ..., ..., i, j, k, study_id),term == 'auditory'
term_images(agg_create_region_overlay(i, j, k, c), term) :- voxel_terms_doc_counts(term, i, j, k, c)
term_images_voxel_count(term, agg_count(c)):-voxel_terms_doc_counts(term, i, j, k, c), c > 5
'''

qw = QueryWidget(nl, query)
qw
# -

from nilearn import plotting

with nl.scope as e:
    nl.execute_datalog_program(query)
    res = nl.solve_all()

plotting.plot_stat_map(res['term_images'].as_pandas_dataframe().iloc[0, 0].spatial_image())


