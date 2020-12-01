# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # NeuroSynth

# +
import sys
import os
import warnings  # type: ignore

warnings.filterwarnings("ignore")

import nibabel as nib
from nilearn.datasets import utils
import numpy as np
import pandas as pd

from typing import Iterable

from neurolang import frontend as fe

from nlweb.viewers.query import QueryWidget


# +
def init_agent():
    nl = fe.NeurolangDL()

    @nl.add_symbol
    def agg_count(x: Iterable) -> int:
        return len(x)

    @nl.add_symbol
    def agg_sum(x: Iterable) -> float:
        return x.sum()

    @nl.add_symbol
    def agg_mean(x: Iterable) -> float:
        return x.mean()

    @nl.add_symbol
    def agg_create_region(x: Iterable, y: Iterable, z: Iterable) -> fe.ExplicitVBR:
        mni_t1 = it.masker.volume
        voxels = nib.affines.apply_affine(np.linalg.inv(mni_t1.affine), np.c_[x, y, z])
        return fe.ExplicitVBR(voxels, mni_t1.affine, image_dim=mni_t1.shape)

    @nl.add_symbol
    def agg_create_region_overlay(
        x: Iterable, y: Iterable, z: Iterable, v: Iterable
    ) -> fe.ExplicitVBR:
        mni_t1 = it.masker.volume
        voxels = nib.affines.apply_affine(np.linalg.inv(mni_t1.affine), np.c_[x, y, z])
        return fe.ExplicitVBROverlay(voxels, mni_t1.affine, v, image_dim=mni_t1.shape)

    return nl


def add_features(nl, features_normalised):
    nl.add_tuple_set(features_normalised.values, name="ns_pmid_term_tfidf")


def add_ns_activations(nl):
    nl.add_tuple_set(
        database[["id", "x", "y", "z", "space"]].values, name="ns_activations"
    )


def add_ns_activations_by_id(nl, study_id_vox_id):
    nl.add_tuple_set(study_id_vox_id, name="ns_activations_by_id")


def add_vox_id_MNI(nl, vox_id_MNI):
    nl.add_tuple_set(vox_id_MNI, name="ns_vox_id_MNI")


def load_database():

    d_neurosynth = utils._get_dataset_dir("neurosynth", data_dir="neurolang_data")

    f_neurosynth = utils._fetch_files(
        d_neurosynth,
        [
            (
                f,
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            )
            for f in ("database.txt", "features.txt")
        ],
        verbose=True,
    )

    database = pd.read_csv(f_neurosynth[0], sep="\t")
    features = pd.read_csv(f_neurosynth[1], sep="\t")

    return database, features


def normalize_features(features):

    features_normalised = features.melt(
        id_vars=features.columns[0],
        var_name="term",
        value_vars=features.columns[1:],
        value_name="tfidf",
    ).query("tfidf > 0")

    return features_normalised


def handle_neurosynth_data():
    nsh = fe.neurosynth_utils.NeuroSynthHandler()
    ns_ds = nsh.ns_load_dataset()
    it = ns_ds.image_table
    vox_ids, study_ids_ix = it.data.nonzero()
    study_ids = ns_ds.image_table.ids[study_ids_ix]
    study_id_vox_id = np.transpose([study_ids, vox_ids])
    masked_ = it.masker.unmask(np.arange(it.data.shape[0]))
    nnz = masked_.nonzero()
    vox_id_MNI = np.c_[
        masked_[nnz].astype(int),
        nib.affines.apply_affine(it.masker.volume.affine, np.transpose(nnz)),
    ]

    return it, study_id_vox_id, vox_id_MNI


# +
# Prepare engine

# prevent stdout to ui in the gallery
with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    #   sys.stdout = devnull

    # Prepare neurosynth data
    database, features = load_database()
    features_normalised = normalize_features(features)
    it, study_id_vox_id, vox_id_MNI = handle_neurosynth_data()

    # Initialize query agent
    nl = init_agent()
    add_features(nl, features_normalised)
    add_ns_activations(nl)
    add_ns_activations_by_id(nl, study_id_vox_id)
    add_vox_id_MNI(nl, vox_id_MNI)

    sys.stdout = old_stdout


# +
# Display query gui
query = """
term_docs(term, pmid) :- ns_pmid_term_tfidf(pmid, term, tfidf),\
    term == 'auditory', tfidf > .003
term_docs(term, pmid) :- ns_pmid_term_tfidf(pmid, term, tfidf),\
    term == 'fear', tfidf > .003

act_term_counts(term, voxid, agg_count(pmid)) :- \
    ns_activations_by_id(pmid, voxid) &\
    term_docs(term, pmid)\

term_counts(term, agg_count(pmid)) :-  \
    ns_pmid_term_tfidf(pmid, term, tfidf) & \
    term_docs(term, pmid)

p_act_given_term(voxid, x, y, z, term, prob) :- \
    act_term_counts(term, voxid, act_term_count) & \
    term_counts(term, term_count) & \
    ns_vox_id_MNI(voxid, x, y, z) & \
    prob == (act_term_count / term_count)


region_prob(agg_create_region_overlay(x, y, z, prob), term) :- \
    p_act_given_term(voxid, x, y, z, term, prob)

thr_prob(agg_create_region(x, y, z), term) :- \
    p_act_given_term(voxid, x, y, z, term, prob) & \
    prob > 0.1
"""


qw = QueryWidget(nl, query)
qw
