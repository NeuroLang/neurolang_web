# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: voila
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
#   voila:
#     template: neurolang
# ---

# %%
# NeuroSynth

# %%
import sys
import os
import warnings  # type: ignore

warnings.filterwarnings("ignore")


from nilearn import datasets, image
import nibabel as nib
import numpy as np
import pandas as pd

from typing import Iterable

from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay
from neurolang import NeurolangPDL

from nlweb.viewers.query import QueryWidget


# %%
def init_agent():

    # Probabilistic Logic Programming in NeuroLang

    nl = NeurolangPDL()

    # Adding new aggregation function to build a region
    @nl.add_symbol
    def agg_count(i: Iterable) -> np.int64:
        return np.int64(len(i))

    # Adding new aggregation function to build a region
    @nl.add_symbol
    def agg_create_region(i: Iterable, j: Iterable, k: Iterable) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBR(voxels, mni_t1_4mm.affine, image_dim=mni_t1_4mm.shape)

    # Adding new aggregation function to build a region overlay
    @nl.add_symbol
    def agg_create_region_overlay(
        i: Iterable, j: Iterable, k: Iterable, p: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBROverlay(
            voxels, mni_t1_4mm.affine, p, image_dim=mni_t1_4mm.shape
        )

    @nl.add_symbol
    def agg_percentile(x: Iterable, q: float) -> float:
        ret = np.percentile(x, q)
        return ret

    return nl


def load_mni_atlas():
    """Load the MNI atlas and resample it to 4mm voxels."""
    data_dir = "neurolang_data"
    mni_t1 = nib.load(datasets.fetch_icbm152_2009(data_dir=data_dir)["t1"])
    return image.resample_img(mni_t1, np.eye(3) * 4)


def load_database(mni_atlas):
    """Load neurosynth database."""
    d_neurosynth = datasets.utils._get_dataset_dir(
        "neurosynth", data_dir="neurolang_data"
    )

    ns_database_fn, ns_features_fn = datasets.utils._fetch_files(
        d_neurosynth,
        [
            (
                "database.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
            (
                "features.txt",
                "https://github.com/neurosynth/neurosynth-data/raw/master/current_data.tar.gz",
                {"uncompress": True},
            ),
        ],
        verbose=0,
    )

    ns_database = pd.read_csv(ns_database_fn, sep="\t")
    ijk_positions = np.round(
        nib.affines.apply_affine(
            np.linalg.inv(mni_atlas.affine),
            ns_database[["x", "y", "z"]].values.astype(float),
        )
    ).astype(int)
    ns_database["i"] = ijk_positions[:, 0]
    ns_database["j"] = ijk_positions[:, 1]
    ns_database["k"] = ijk_positions[:, 2]

    ns_features = pd.read_csv(ns_features_fn, sep="\t")

    ns_database = ns_database.query("space == 'MNI'")[
        ["x", "y", "z", "i", "j", "k", "id"]
    ].rename(columns={"id": "pmid"})

    return ns_database, ns_features


def add_activations(nl, ns_database):
    nl.add_tuple_set(ns_database, name="activations")


def add_terms(nl, ns_features):
    ns_terms = pd.melt(
        ns_features, var_name="term", id_vars="pmid", value_name="TfIdf"
    ).query("TfIdf > 1e-3")[["pmid", "term", "TfIdf"]]
    nl.add_tuple_set(ns_terms, name="terms")


def add_docs(nl, ns_features):
    ns_docs = ns_features[["pmid"]].drop_duplicates()
    nl.add_uniform_probabilistic_choice_over_set(ns_docs.values, name="docs")


# %%
# Prepare engine

# prevent stdout to ui in the gallery
with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull

    mni_t1_4mm = load_mni_atlas()
    # Prepare neurosynth data
    ns_database, ns_features = load_database(mni_t1_4mm)

    # Initialize query agent
    nl = init_agent()

    # Loading the database
    add_activations(nl, ns_database)
    add_terms(nl, ns_features)
    add_docs(nl, ns_features)

    sys.stdout = old_stdout

# %%
# Display query gui
query = r"""
activation(i, j, k) :- activations(..., ..., ..., i, j, k, study_id), docs(study_id)
term_(term) :- terms(study_id, term, tfidf), tfidf > 0.01, docs(study_id)
activation_given_term_marginal(i, j, k, PROB(i, j, k)) :- activation(i,j,k) // (term_(term), term == 'auditory')
activation_given_term_image(agg_create_region_overlay(i, j, k, p)) :- activation_given_term_marginal(i,j,k,p)
"""

qw = QueryWidget(nl, query)
qw

# %%
