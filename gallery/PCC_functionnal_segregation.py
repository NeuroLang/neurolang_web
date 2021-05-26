# -*- coding: utf-8 -*-
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

# %% [markdown]
# # Using Probabilistic First-Order Logic in NeuroLang To Segregate Dorsal and Ventral Posterior Cingulate Cortex (PCC)

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import nilearn.datasets
from neurolang.frontend import NeurolangPDL, ExplicitVBR, ExplicitVBROverlay

from gallery import data_utils

# %%
data_dir = Path("neurolang_data")

# %%
def init_frontend():
    nl = NeurolangPDL()

    # Function to check if a string starts with a given prefix
    @nl.add_symbol
    def startswith(prefix: str, s: str) -> bool:
        """Describe the prefix of string `s`.

        Parameters
        ----------
        prefix : str
            prefix to query.
        s : str
            string to check whether its
            prefixed by `s`.

        Returns
        -------
        bool
            whether `s` is prefixed by
            `prefix`.
        """
        return s.startswith(prefix)

    # Log-odds Ratio
    @nl.add_symbol
    def log_odds(p, p0):
        """Compute Log-Odds Ratio.

        Parameters
        ----------
        p, p0 : Float. Probabilities of two events 

        Returns
        -------
        logodds : Float. Log-odds Ratio 
        """

        logodds = np.log((p / (1 - p)) / (p0 / (1 - p0)))
        return logodds

    # Aggregation function to build a region
    @nl.add_symbol
    def agg_create_region(
        i: Iterable, j: Iterable, k: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBR(voxels, mni_mask.affine, image_dim=mni_mask.shape)

    # Aggregation function to build a region overlay
    @nl.add_symbol
    def agg_create_region_overlay(
        i: Iterable, j: Iterable, k: Iterable, p: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBROverlay(
            voxels, mni_mask.affine, p, image_dim=mni_mask.shape
        )

    return nl


# %% [markdown]
"""
## Fetch the data from atlases
We start by loading the data we need from available atlases.

We use the Neurosynth CBMA database (Yarkoni et al., 2011), consisting of 14,371 studies, 
its associated `v5-topics-200` topic model (Poldrack et al., 2012), and the DiFuMo functional atlas.
"""

# %%
# MNI Template
resolution = 3
mni_mask = data_utils.load_mni_atlas(
    data_dir=data_dir, resolution=resolution, key="t1"
)

# %%
# Difumo 128 Regions Atlas
coord_type = "ijk"
n_components = 128
region_voxels, difumo_meta = data_utils.fetch_difumo(
    data_dir=data_dir,
    mask=mni_mask,
    coord_type=coord_type,
    n_components=n_components,
)

difumo_img = nilearn.datasets.fetch_atlas_difumo(
    dimension=n_components, data_dir=str(data_dir)
).maps

# %%
# NeuroSynth database
term_in_study, peak_reported, study_ids = data_utils.fetch_neurosynth(
    data_dir=data_dir
)
ijk_positions = np.round(
    data_utils.xyz_to_ijk(
        peak_reported[["x", "y", "z"]].values.astype(float), mni_mask
    )
)
peak_reported["i"] = ijk_positions[:, 0]
peak_reported["j"] = ijk_positions[:, 1]
peak_reported["k"] = ijk_positions[:, 2]

# %%
nl = init_frontend()

# %% [markdown]
"""
## Load the data into the Neurolang engine

We load the data into tables in the Neurolang engine :

* **PeakReported** is a relation, or tuple, that includes the peak coordinates (x, y, z) reported in each study.
* **RegionVoxel** is a relation, or tuple, that includes the label of each region in the DiFuMo atlas and the coordinates of each voxel within each region in voxel space (i, j, k).
* **Study** is a relation, or tuple, with one variable corresponding to the $\textit{id}$ of each study.
* **SelectedStudy** annotates each study with a probability equal to 1/N of it being chosen out of the whole dataset of size N.
"""
# %%
def load_studies(
    nl, region_voxels, peak_reported, study_ids,
):
    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_tuple_set(region_voxels, name="RegionVoxel")
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )


# %%
load_studies(nl, region_voxels, peak_reported, study_ids)

# %% [markdown]
"""
We also select 16 Topics of Interest from Version 5 of Neurosynth's 200 Topics List, which we then add as a **TopicInStudy** 
relation in the Neurolang engine
"""

# %%
def load_topics(nl):
    topics_to_keep = [
        182,
        90,
        30,
        28,
        145,
        154,
        137,
        159,
        52,
        3,
        179,
        8,
        186,
        197,
        66,
        92,
    ]

    labels = [
        "subjective experience",
        "mental rotation",
        "mental time travel",
        "episodic memory",
        "mentalizing",
        "social cognition",
        "emotion regulation",
        "response inhibition",
        "cognitive control",
        "task switching",
        "working memory",
        "attention allocation",
        "decision making",
        "reward/motivation",
        "emotional valence",
        "central executive",
    ]

    n_topics = 200
    topic_association = data_utils.fetch_neurosynth_topic_associations(
        n_topics, data_dir=data_dir, topics_to_keep=topics_to_keep, labels=labels
    )

    nl.add_probabilistic_facts_from_tuples(
        topic_association[["prob", "topic", "study_id"]].itertuples(index=False),
        name="TopicInStudy",
    )

# %%
load_topics(nl)

# %% [markdown]
"""
## Formulate a NeuroLang query that Extracts the DiFuMo regions in the PCC

The following query reads: A region is in PCC if it has voxels (RegionVoxel) in the DiFuMo atlas 
and its label starts with Posterior cingulate.

```python
PCCRegion(difumo_label, i, j, k) :- RegionVoxel(difumo_label, i, j, k) & startswith("Posterior cingulate", difumo_label)
```
"""

# %%
pcc_query = r"""
PCCRegion(difumo_label, i, j, k) :- RegionVoxel(difumo_label, i, j, k) & startswith("Posterior cingulate", difumo_label)
ans(difumo_label, i, j, k) :- PCCRegion(difumo_label, i, j, k)
"""

# %%
pcc_region = nl.execute_datalog_program(pcc_query).as_pandas_dataframe()
pcc_region

# %% [markdown]
"""
## Categorize the PCC regions into Dorsal (dPCC) and Ventral (vPCC)

Once the PCC regions are identified, we categorize them into either Dorsal and Ventral, which allows us to define more
relations in Neurolang:

* **PCClabel** is a relation, or tuple, that stores the labels dPCC or vPCC.
* **RegionLabel**  is a relation, or tuple, that stores the labels of DiFuMo PCC regions.
* **RegionOfInterest** is a relation, or tuple, that stores the label of each DiFuMo region within the PCC, its voxel coordinates (i, j, k) and a column describing its position in the PCC (dPCC vs. vPCC).
"""

# %%
dpcc = np.where(
    (
        pcc_region["difumo_label"].values
        == "Posterior cingulate cortex antero-inferior"
    )
    | (
        pcc_region["difumo_label"].values
        == "Posterior cingulate cortex superior"
    )
)

vpcc = np.where(
    (pcc_region["difumo_label"].values == "Posterior cingulate cortex")
    | (
        pcc_region["difumo_label"].values
        == "Posterior cingulate cortex inferior"
    )
)

pcc_region["pcc_label"] = ""
pcc_region["pcc_label"].iloc[dpcc] = "dPCC"
pcc_region["pcc_label"].iloc[vpcc] = "vPCC"
pcc_region

# %%
nl.add_tuple_set([('dPCC',), ('vPCC',)],name="PCCLabel")
nl.add_tuple_set(np.unique(pcc_region['difumo_label'].values), name="RegionLabel")
nl.add_tuple_set(pcc_region, name="RegionOfInterest")


# %%
