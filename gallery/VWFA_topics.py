# -*- coding: utf-8 -*-
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

# %% [markdown]
# # Meta-Analysing the Role of the Visual Word-Form Area in Attention Circuitry

# %% [markdown]
"""
Recently (Chen et al., 2019) provided evidence that the visual word-form area (VWFA) was part of attention circuitry, 
through an analysis of high-resolution multimodal imaging data from a Human Connectome Project cohort (Van Essen et al., 2013). 
Can this relationship be identified solely from a meta-analysis of past studies that have reported activations in the left 
ventral occipito-temporal cortex without necessarily identifying it as the VWFA?
"""

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable

import numpy as np
from neurolang.frontend import NeurolangPDL

from gallery import data_utils

# %%
data_dir = Path("neurolang_data")

# %%
def init_frontend():
    nl = NeurolangPDL()

    nl.add_symbol(
        np.log, name="log", type_=Callable[[float], float],
    )

    return nl


# %% [markdown]
"""
We use the Neurosynth CBMA database (Yarkoni et al., 2011), consisting of 14,371 studies, 
and its associated `v5-topics-100` topic model (Poldrack et al., 2012), encoded in the `TopicAssociation` database table. 
"""

# %%
def load_studies(
    nl,
    topic_association,
    peak_reported,
    study_ids,
    split_id: int = 42,
    subsample_proportion: float = 0.6,
):
    n_studies_selected = int(len(study_ids) * subsample_proportion)
    np.random.seed(split_id)
    study_ids = study_ids.sample(n_studies_selected)
    peak_reported = peak_reported.loc[
        peak_reported.study_id.isin(study_ids.study_id)
    ]
    topic_association = topic_association.loc[
        topic_association.study_id.isin(study_ids.study_id)
    ]
    nl.add_probabilistic_facts_from_tuples(
        set(topic_association.itertuples(index=False, name=None)),
        name="TopicAssociation",
    )
    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )
    nl.add_tuple_set(study_ids, name="Study")


# %% [markdown]
"""
To define the VWFA, fronto-parietal attention network and ‘language’ network, 
we use the same seed locations as in (Chen et al., 2019), 
stored in a `RegionSeedVoxel` database table that contains a row (x, y, z, r) for each region r’s seed voxel 
located at MNI coordinates (x, y, z). 

A database table `RegionInNetwork` contains rows (n, r) for each region r belonging to network n. 
"""

# %%
def load_attention_language_networks(nl):
    nl.add_tuple_set([("attention",), ("language",)], name="Network")
    nl.add_tuple_set(
        {
            ("FEF", "attention"),
            ("aIPS", "attention"),
            ("pIPS", "attention"),
            ("MT+", "attention"),
            ("IFG", "language"),
            ("SMG", "language"),
            ("AG", "language"),
            ("ITG", "language"),
            ("aSTS", "language"),
            ("mSTS", "language"),
            ("pSTS", "language"),
        },
        name="RegionInNetwork",
    )
    nl.add_tuple_set(
        {
            # vwfa
            ("VWFA", -45, -57, -12),
            # attention regions
            ("FEF", -26, -5, 50),
            ("MT+", -45, -71, -1),
            ("aIPS", -25, -62, 51),
            ("pIPS", -25, -69, 34),
            # language regions
            ("IFG", -53, 27, 16),
            ("SMG", -56, -43, 31),
            ("AG", -49, -57, 28),
            ("ITG", -61, -33, -15),
            ("aSTS", -54, -9, -20),
            ("mSTS", -53, -18, -10),
            ("pSTS", -52, -40, 5),
        },
        name="RegionSeedVoxel",
    )


# %%
resolution = 3
mni_mask = data_utils.load_mni_atlas(data_dir=data_dir, resolution=resolution)

# %%
n_topics = 100
topic_association = data_utils.fetch_neurosynth_topic_associations(n_topics, data_dir=data_dir)

# %%
_, peak_reported, study_ids = data_utils.fetch_neurosynth(tfidf_threshold=1e-2, data_dir=data_dir)

# %%
nl = init_frontend()

# %%
load_studies(nl, topic_association, peak_reported, study_ids)
load_attention_language_networks(nl)

# %% [markdown]
"""
We formulate NeuroLang queries that infer the most probable topic associations among studies that 
report activations close to the VWFA region, while simultaneously reporting activations within the 
fronto-parietal attention network, but not reporting activations within the ‘language’ network.

By excluding studies that report activations within the language network, we maintain the focus of 
the meta-analysis on studies that might be studying the attention circuitry, while still reporting activations within the VWFA.
"""

# %% [markdown]
"""
A brain region is considered to be reported by a study if it reports a peak activation within 10mm 
of the region seed voxel’s location. A network is considered to be reported by a study if it reports 
one of the network’s regions, based on the previous definition.

In NeuroLang, this is expressed with the following rules, where `EUCLIDEAN` is a built-in function that 
calculates the Euclidean distance between two coordinates in MNI space :
```python
RegionReported(r, s) :- PeakReported(x1, y1, z1, s) & RegionSeedVoxel(r, x2, y2, z2) & (d == EUCLIDEAN(x1, y1, z1, x2, y2, z2)) & (d < 10.0)
NetworkReported(n, s) :- RegionReported(r, s) & RegionInNetwork(r, n)
````
"""

# %%
query = r"""
RegionReported(r, s) :- PeakReported(x1, y1, z1, s) & RegionSeedVoxel(r, x2, y2, z2) & (d == EUCLIDEAN(x1, y1, z1, x2, y2, z2)) & (d < 10.0)
NetworkReported(n, s) :- RegionReported(r, s) & RegionInNetwork(r, n)
"""

# %% [markdown]
"""
Finally, to test our hypothesis, we use the following probability encoding rule
which calculates the probability of finding an association with topic *t* among studies that 
report the activation of the **VWFA**, the activation of the network *n*, but do not report the 
activation of any other network *n2*, where *n2 ̸= n*. 

```python
StudyMatchingNetworkQuery(s, n) :- RegionReported("VWFA", s) & NetworkReported(n, s) & exists(n2; ((n2 != n) & NetworkReported(n2, s) & Network(n2)))
PositiveReverseInferenceSegregationQuery(t, n, PROB(t, n)) :- (TopicAssociation(t, s) & SelectedStudy(s)) // (StudyMatchingNetworkQuery(s, n) & SelectedStudy(s))
```

Because only two networks, `language` and `attention`, 
are present in the `Network` table, this rule simultaneously calculates the probabilities for each pair of networks, 
including one while segregating the other.
"""

# %%
query += r"""
StudyMatchingNetworkQuery(s, n) :- RegionReported("VWFA", s) & NetworkReported(n, s) & exists(n2; ((n2 != n) & NetworkReported(n2, s) & Network(n2)))
PositiveReverseInferenceSegregationQuery(t, n, PROB(t, n)) :- (TopicAssociation(t, s) & SelectedStudy(s)) // (StudyMatchingNetworkQuery(s, n) & SelectedStudy(s))
"""

# %%
query += r"""
StudyNotMatchingSegregationQuery(s, n) :- ~StudyMatchingNetworkQuery(s, n) & Study(s) & Network(n)
NegativeReverseInferenceSegregationQuery(t, n, PROB(t, n)) :- (TopicAssociation(t, s) & SelectedStudy(s)) // (StudyNotMatchingSegregationQuery(s, n) & SelectedStudy(s))
MarginalTopicAssociation(t, PROB(t)) :- SelectedStudy(s) & TopicAssociation(t, s)
CountStudies(count(s)) :- Study(s)
CountStudiesWithTopic(t, c) :- MarginalTopicAssociation(t, prob) & (c == prob * N) & CountStudies(N)
CountStudiesMatchingQuery(n, count(s)) :- StudyMatchingNetworkQuery(s, n)
JointProb(t, n, PROB(t, n)) :- TopicAssociation(t, s) & StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)
CountStudiesMatchingQueryWithTopic(t, n, c) :- JointProb(t, n, p) & CountStudies(N) & (c == N * p)
LikelihoodRatio(topic, network, p1, p0, llr, m, n, k) :- PositiveReverseInferenceSegregationQuery(topic, network, p1) & NegativeReverseInferenceSegregationQuery(topic, network, p0) & MarginalTopicAssociation(topic, p) & CountStudies(N) & CountStudiesWithTopic(topic, m) & CountStudiesMatchingQuery(network, n) & CountStudiesMatchingQueryWithTopic(topic, network, k) & ( llr == ( k * log(p1) + ((n - k) * log(1 - p1) + ((m - k) * log(p0) + (((N - n) - (m + k)) * log(1 - p0))))) - ( k * log(p) + ((n - k) * log(1 - p) + ((m - k) * log(p) + (((N - n) - (m + k)) * log(1 - p))))))
ans(topic, network, pTgQ, pTgNotQ, llr, nb_studies_associated_with_topic, nb_studies_matching_segregation_query, nb_studies_both_associated_with_topic_and_matching_query) :- LikelihoodRatio(topic, network, pTgQ, pTgNotQ, llr, nb_studies_associated_with_topic, nb_studies_matching_segregation_query, nb_studies_both_associated_with_topic_and_matching_query)
"""

# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, query)
qw

# %%
