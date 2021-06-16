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
# # Using an Ontology for Synonyms meta-analysis

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable, Iterable

import nibabel
import numpy as np
import pandas as pd
from neurolang.frontend import NeurolangPDL, ExplicitVBR, ExplicitVBROverlay

from nlweb import data_utils

# %%
data_dir = Path("neurolang_data")

# %%
def init_frontend():
    nl = NeurolangPDL()

    @nl.add_symbol
    def first_word(name: str) -> str:
        return name.split(" ")[0]

    @nl.add_symbol
    def mean(iterable: Iterable) -> float:
        return np.mean(iterable)

    @nl.add_symbol
    def std(iterable: Iterable) -> float:
        return np.std(iterable)

    @nl.add_symbol
    def agg_create_region_overlay_MNI(
        x: Iterable, y: Iterable, z: Iterable, p: Iterable
    ) -> ExplicitVBR:
        voxels = nibabel.affines.apply_affine(
            np.linalg.inv(mni_mask.affine), np.c_[x, y, z]
        )
        return ExplicitVBROverlay(voxels, mni_mask.affine, p, image_dim=mni_mask.shape)

    nl.add_symbol(np.exp, name="exp", type_=Callable[[float], float])
    nl.add_symbol(nl.new_symbol(name="neurolang:label"), name="label")
    nl.add_symbol(nl.new_symbol(name="neurolang:related"), name="related")
    nl.add_symbol(nl.new_symbol(name="neurolang:altLabel"), name="altLabel")
    nl.add_symbol(nl.new_symbol(name="neurolang:subClassOf"), name="subclass_of")

    return nl


# %% [markdown]
"""
### Fetch the data
We start by loading the data we need from available atlases.

We use the Neurosynth CBMA database (Yarkoni et al., 2011), consisting of 14,371 studies.
We also use biological ontologies which contain biological concepts and the relations between them.

In this example, we use the Cognitive Atlas ontology, but others can be used as well, such as the Interlinking Ontology 
for Biological Concepts (IOBC) which contains approximately 80,000 biological concepts.
"""

# %%
# MNI Template
resolution = 2
mni_mask = data_utils.load_mni_atlas(data_dir=data_dir, resolution=resolution)

# %%
# IOBC Ontology
iobc = data_utils.fetch_iobc_ontology(data_dir=data_dir)
cogat = data_utils.fetch_cogat_ontology(data_dir=data_dir)

# %%
# NeuroSynth database
tfidf_threshold = 1e-3
term_in_study, peak_reported, study_ids = data_utils.fetch_neurosynth(
    data_dir=data_dir, tfidf_threshold=tfidf_threshold
)


# %%
nl = init_frontend()

# %% [markdown]
"""
### Load the data into the Neurolang engine

We load the data into tables in the Neurolang engine :

* **PeakReported** is a relation, or tuple, that includes the peak coordinates (x, y, z) reported in each study.
* **RegionVoxel** is a relation, or tuple, that includes the label of each region in the DiFuMo atlas and the coordinates of each voxel within each region in voxel space (i, j, k).
* **Study** is a relation, or tuple, with one variable corresponding to the $\textit{id}$ of each study.
* **SelectedStudy** annotates each study with a probability equal to 1/N of it being chosen out of the whole dataset of size N.

We also generate random splits of the Neurosynth dataset. We select a small number of splits to be able to compute the results in 
reasonable time, but this parameter can be increased. The ids of the studies in each splits are stored in the **StudyFolds** relation.
"""
# %%
from sklearn.model_selection import KFold


def load_studies(
    nl,
    mni_mask,
    term_in_study,
    peak_reported,
    study_ids,
    n_splits,
):
    nl.add_tuple_set(term_in_study[["term", "study_id"]], name="TermInStudy")
    nl.add_tuple_set(peak_reported, name="FocusReported")
    nl.add_uniform_probabilistic_choice_over_set(study_ids, name="SelectedStudy")
    nl.add_tuple_set(study_ids, name="Study")

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    study_id_folds = pd.concat(
        study_ids.iloc[train].assign(fold=[i] * len(train))
        for i, (train, _) in enumerate(kfold.split(study_ids))
    )
    nl.add_tuple_set(study_id_folds, name="StudyFolds")

    nl.add_tuple_set(
        nibabel.affines.apply_affine(
            mni_mask.affine, np.transpose(mni_mask.get_fdata().nonzero())
        ),
        name="Voxel",
    )


# %%
n_splits = 5
load_studies(nl, mni_mask, term_in_study, peak_reported, study_ids, n_splits=n_splits)

# %%
nl.load_ontology(cogat)

# %% [markdown]
"""
### Write a NeuroLang program

We'll start by looking at synonyms **Pain**, **Noxious**, **Nociceptive**.

The first rule of the program, 
```python
RelatedBiostimulationTerm(word) :- (word == 'pain')
```

sets the word we're looking for, while the second rule,

```python
RelatedBiostimulationTerm(alternative_names) :- subclass_of(biostimulation_subclass, '200906066643737725') & label(pain_entity, 'Pain') &  related(pain_entity, biostimulation_subclass) & altLabel(biostimulation_subclass, alternative_names)
```

finds alternative names for this term.

Try to remove the first rule to look at the mean activation values for the synonyms without the word **pain**.
"""

# %%
synonyms_query = r"""
RelatedBiostimulationTerm(word) :- (word == 'pain')
RelatedBiostimulationTerm(alternative_names) :- subclass_of(biostimulation_subclass, '200906066643737725') & label(pain_entity, 'Pain') &  related(pain_entity, biostimulation_subclass) & altLabel(biostimulation_subclass, alternative_names)
Synonym(short_name) :- (short_name == first_word(alternative_names)) & RelatedBiostimulationTerm(alternative_names)
FilteredBySynonym(t, s) :- TermInStudy(t, s) & Synonym(t)
VoxelReported(x, y, z, s) :-  FocusReported(x2, y2, z2, s) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 1)
Result(x, y, z, fold, PROB(x, y, z, fold)) :- VoxelReported(x, y, z, s) // ( SelectedStudy(s) & FilteredBySynonym(t, s) & StudyFolds(s, fold) )
ResultMean(x, y, z, mean(p)) :- Result(x, y, z, fold, p)
ResultStd(x, y, z, std(p)) :- Result(x, y, z, fold, p)
ResultSummaryStats(x, y, z, prob_mean, prob_std) :- ResultMean(x, y, z, prob_mean) & ResultStd(x, y, z, prob_std)
VoxelActivationImg(agg_create_region_overlay_MNI(x, y, z, p)) :- ResultMean(x, y, z, p)"""


# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, synonyms_query)
qw

# %% [markdown]
"""
### Single term analysis

We can also run single term analysis by fixing the term in the program without using the ontology to select synonyms.
In the query below, we look at the activation map for the term **noxious**. Change the term defined in the first rule
to look at the activation map for other terms.
"""

# %%
single_term_query = r"""
FilteredByTerm(t, s) :- TermInStudy(t, s) & (t == 'noxious')
VoxelReported(x, y, z, s) :-  FocusReported(x2, y2, z2, s) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 1)
Result(x, y, z, fold, PROB(x, y, z, fold)) :- VoxelReported(x, y, z, s) // ( SelectedStudy(s) & FilteredByTerm(t, s) & StudyFolds(s, fold) )
ResultMean(x, y, z, mean(p)) :- Result(x, y, z, fold, p)
ResultStd(x, y, z, std(p)) :- Result(x, y, z, fold, p)
ResultSummaryStats(x, y, z, prob_mean, prob_std) :- ResultMean(x, y, z, prob_mean) & ResultStd(x, y, z, prob_std)
VoxelActivationImg(agg_create_region_overlay_MNI(x, y, z, p)) :- ResultMean(x, y, z, p)
"""

# %%
qw = QueryWidget(nl, single_term_query)
qw

# %%
