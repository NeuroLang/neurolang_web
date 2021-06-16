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
"""
# Meta-Analysis of the Neurosynth CBMA database using Neurolang

Meta-analysis is an essential part of cognitive neuroscience research for aggregating results from individual 
neuroimaging studies to derive consistent and novel patterns of brain-behavior relationships. However, existing 
tools for automated meta-analysis are limited to expressing simple hypotheses and addressing a restricted range 
of neuroscience questions. Neurolang expands the scope of automated meta-analysis with a domain-specific language 
to express and test meta-analytic hypotheses using probabilistic logic programming.

NeuroLang can be applied to the resolution of term-based and coactivation queries on CBMA databases, which are the 
most common types of meta-analyses conducted in the literature. A *term-based* query derives an activation pattern associated 
with a term of interest that relates to psychological concepts or cognitive process. A *coactivation* query delineates a brain 
activation pattern that comprises spatially distant brain regions, putatively forming a large-scale functional network. 
Answering these queries is typically done using tools like Neurosynth [[1]](#1).
"""

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import nibabel
from neurolang import NeurolangPDL
from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay
from nlweb import data_utils

# %%
data_dir = Path("neurolang_data")

# %%
def init_frontend():
    """
    Create a Neurolang Probabilistic engine and add some aggregation methods.

    Returns
    -------
    NeurolangPDL
        the Neurolang engine
    """
    nl = NeurolangPDL()

    @nl.add_symbol
    def agg_count(i: Iterable) -> np.int64:
        return np.int64(len(i))

    @nl.add_symbol
    def agg_create_region(
        i: Iterable, j: Iterable, k: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBR(voxels, mni_mask.affine, image_dim=mni_mask.shape)

    @nl.add_symbol
    def agg_create_region_overlay(
        i: Iterable, j: Iterable, k: Iterable, p: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBROverlay(
            voxels, mni_mask.affine, p, image_dim=mni_mask.shape
        )

    @nl.add_symbol
    def agg_percentile(x: Iterable, q: float) -> float:
        ret = np.percentile(x, q)
        return ret

    nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])

    return nl


# %%
# MNI Template with resolution 4mm
resolution = 2
mni_mask = data_utils.load_mni_atlas(
    data_dir=data_dir, resolution=resolution
)

# %%
# Load NeuroSynth database
tfidf_threshold = 1e-2
term_in_study, peak_reported, study_ids = data_utils.fetch_neurosynth(
    data_dir=data_dir, tfidf_threshold=tfidf_threshold
)
ijk_positions = np.round(
    data_utils.xyz_to_ijk(
        peak_reported[["x", "y", "z"]].values.astype(float), mni_mask
    )
)
peak_reported["i"] = ijk_positions[:, 0]
peak_reported["j"] = ijk_positions[:, 1]
peak_reported["k"] = ijk_positions[:, 2]
peak_reported = peak_reported[["i", "j", "k", "study_id"]]

# %%
nl = init_frontend()


# %% [markdown]
"""
### Load the data into the Neurolang engine

We load the data from the Neurosynth database into tables in the Neurolang engine :

* **PeakReported** is a relation, or tuple, that includes the peak coordinates (x, y, z) reported in each study.
* **Study** is a relation, or tuple, with one variable corresponding to the *id* of each study.
* **SelectedStudy** annotates each study with a probability equal to 1/N of it being chosen out of the whole dataset of size N.
* **TermInStudyTFIDF** is a relation, or tuple, that includes the terms reported in each study with their tfidf.
"""

# %%
def load_database(
    nl,
    mni_mask,
    term_in_study,
    peak_reported,
    study_ids,
):
    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_tuple_set(term_in_study, name="TermInStudyTFIDF")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )
    nl.add_tuple_set(
         np.hstack(
             np.meshgrid(
                 *(np.arange(0, dim) for dim in mni_mask.get_fdata().shape)
             )
         )
         .swapaxes(0, 1)
         .reshape(3, -1)
         .T,
         name="Voxel",
    )


# %%
load_database(nl, mni_mask, term_in_study, peak_reported, study_ids)

# %% [markdown]
"""
## Running a meta-analysis program

A core goal of meta-analyses is to identify brain regions that are preferentially activated in experiments studying a 
psychological or cognitive process, relative to the rest of the brain. As an example, a meta-analysis of studies on emotion 
should find that the amygdala is one of the regions that is the most likely to be reported by this set of studies. 
In other words, the meta-analysis should find that there is an association between studies of emotion and a neural response 
in the amygdala.


Several models exist for computing probabilistic brain maps from a set of studies associated with a given term of interest (*‘emotion’*, in this example). All provide an estimate of the probability that a voxel gets reported by studies associated with the term. We explore and implement one of them in this example: **multilevel kernel density analysis (MKDA)**.

### Smoothing the spatial prior

An important point should be made first, regarding the differences between approaches to combining the peak activation coordinates that are reported by neuroimaging studies. Peak activations are 3-dimensional points in the brain that live in a standardised stereotactic coordinate system, and that can be seen as a sparse representation of the statistical map obtained from an experimental study of neuroimaging signals. To account for the noise and variability of the *exact location* of these peaks, meta-analysis models often assume a neighborhood around peaks to be reported, or to have a probability of being reported. The definition of this spatial neighborhood is an *a priori* of the meta-analysis, sometimes called a spatial prior. This process, applied to the map of each study separately, is called *smoothing*.
A simple but common spatial smoothing approach is to consider all voxels within a sphere centered at a reported peak location’s coordinates to also be reported by the study. MKDA obtains one such binary pattern for each study, and aggregates them as frequencies at which each voxel is reported by studies selected for a meta-analysis. In plain English, ‘a voxel at location (*i, j, k*) is reported by a study *s* whenever *s* reports a peak activation within 10mm of that voxel’. In NeuroLang, this sentence is encoded by the following **VoxelReported** rule
```python
  VoxelReported(x, y, z, s) :-
      PeakReported(x2, y2, z2, s) & Voxel(x, y, z)
      & d = euclidean_distance(x, y, z, x2, y2, z2) & d < 10
```
where the pairwise Euclidean distance between pairs of voxels is calculated using a built-in `euclidean_distance` function (**EUCLIDEAN**). The lower bound d < 10 defines a 10mm radius ball around each peak location. An MKDA meta-analytic map is obtained by estimating the conditional probability that each voxel gets reported by studies associated with a given term of interest (e.g. *‘emotion’*).

---
**NOTE**

In the query below, we set the lower bound d < 1 since computing all the voxels in a larger radius around each peak location is resource intensive, but this parameter can be tweaked to add a bit more smoothing.

---

### Estimate the probability of each voxel being reported

The distribution of the program’s outputs can then be used to estimate the probability of each
voxel being reported. The result of solving a conditional probabilistic rule
```python
  ActivationGivenTerm(i, j, k, PROB(i, j, k)) :-
      VoxelReported(i, j, k, s) & SelectedStudy(s)
      // TermAssociation("emotion", s) & SelectedStudy(s)
```
will contain a probability for each voxel of the brain, based on the frequency at which it is reported by studies within the database. The **//** operator represents a *probabilistic conditioning*.

---
**NOTE**

In the query below, we split the conditional probabilistic rule into sub rules for readability.

---
"""

# %%
# Display query gui
query = """
tfidf :: TermInStudy(term, study) :- TermInStudyTFIDF(term, tfidf, study)
VoxelReported (i, j, k, study) :- PeakReported(i2, j2, k2, study) & Voxel(i, j, k) & (d == EUCLIDEAN(i, j, k, i2, j2, k2)) & (d < 1)
TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
Activation(i, j, k) :- SelectedStudy(s) & VoxelReported(i, j, k, s)
ActivationGivenTerm(i, j, k, PROB(i, j, k)) :- Activation(i, j, k) // TermAssociation("emotion")
ActivationGivenTermImage(agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, p)"""

# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(
    nl,
    query,
)
qw


# %% [markdown]
"""
### References
<a id="1">[1]</a>
T. Yarkoni, R. A. Poldrack, T. E. Nichols, D. C. Van Essen, T. D. Wager, Large-scale
automated synthesis of human functional neuroimaging data. *Nature Methods 8*, 665–670."""
