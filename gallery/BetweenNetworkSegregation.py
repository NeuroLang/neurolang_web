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
# # Between-Network Segregation: Reverse Inference of Brain Network Function

# %% [markdown]
"""
In this example, we formulate a reverse inference *segregation query* that derives the probability of a psychological 
topic being present given knowledge of activation in a particular brain network, with an additional constraint that *no 
activation in other networks is reported*. The added constraint can be readily expressed in NeuroLang and aims to functionally 
segregate brain networks in order to assess their functional specializations with greater specificity. 
In other words, the goal of this example is to show that a reverse inference segregation query can identify which network’s 
activation provides more evidence in favor of the engagement of a particular cognitive process.
"""

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import nibabel
from neurolang.frontend import NeurolangPDL

from gallery import data_utils

# %%
data_dir = Path("neurolang_data")

# %%
def init_frontend():
    nl = NeurolangPDL()

    return nl


# %% [markdown]
"""
The networks included in this example are the default mode network DMN, frontoparietal control network FPCN 
and the dorsal attention network DAN. These networks exhibit competitive and cooperative coupling dynamics 
in support of a wide array of internally and externally directed mental functions [[1]](#1). Yet, each one of them 
is believed to be specialized for certain broad cognitive processes ([[2]](#2), [[1]](#1), [[3]](#3)). The FPCN contributes to a wide 
variety of tasks by engaging top-down cognitive control processes; the DAN is concerned with orienting attention 
towards a particular stimuli, location, or task; and the DMN is involved in higher-order self-referential, social 
and affective functions.

Therefore, using a segregation query, we can identify the functional specializations of these networks, 
reflecting upon the general understanding of their roles in the literature.
"""

# %%
def load_topics(n_topics: int = 200):
    topics_to_keep = [3, 147, 157, 118, 187]

    labels = [
        "Memory Retrieval",
        "Task Set Switching",
        "Working Memory",
        "Decision Making",
        "Semantics",
    ]

    topic_association = data_utils.fetch_neurosynth_topic_associations(
        n_topics,
        data_dir=data_dir,
        topics_to_keep=topics_to_keep,
        labels=labels,
        version="v4",
    )
    return topic_association[["prob", "topic", "study_id"]]


# %%
n_topics = 200
topic_association = load_topics(n_topics=n_topics)

# %%
def load_studies(
    nl,
    topic_association,
    term_data,
    peak_reported,
    study_ids,
    split_id: int = 42,
    subsample_proportion: float = 0.6,
):
    n_studies_selected = int(len(study_ids) * subsample_proportion)
    np.random.seed(split_id)
    study_ids = study_ids.sample(n_studies_selected)
    term_data = term_data.loc[term_data.study_id.isin(study_ids.study_id)]
    peak_reported = peak_reported.loc[
        peak_reported.study_id.isin(study_ids.study_id)
    ]

    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_tuple_set(
        term_data[["tfidf", "term", "study_id"]], name="NeuroQueryTFIDF"
    )
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )


# %%
def load_voxels(nl, region_voxels, difumo_meta):
    nl.add_tuple_set(region_voxels, name="RegionVoxel")
    nl.add_tuple_set({("ContA",), ("ContB",)}, name="Network")
    nl.add_tuple_set(
        set(
            (row["Yeo_networks17"], row["Difumo_names"])
            for _, row in difumo_meta.iterrows()
            if row["Yeo_networks17"] in ("ContA", "ContB")
        ),
        name="NetworkRegion",
    )


# %%
resolution = 3
mni_mask = data_utils.load_mni_atlas(data_dir=data_dir, resolution=resolution)


# %%
coord_type = "ijk"
term_data, peak_reported, study_ids = data_utils.fetch_neuroquery(
    mask=mni_mask, coord_type=coord_type, data_dir=data_dir
)

# %%
n_difumo_components = 256
region_voxels, difumo_meta = data_utils.fetch_difumo(
    mask=mni_mask, coord_type=coord_type, n_components=n_difumo_components, data_dir=data_dir
)


# %%
nl = init_frontend()

# %%
load_studies(nl, topic_association, peak_reported, study_ids)
load_voxels(nl, region_voxels, mni_mask)

# %% [markdown]
"""
We assume a DiFuMo-256 component *r* to be reported by a study *s* whenever a peak activation is reported by the study 
within that region. In NeuroLang, this is expressed by the following logic implication rule :

```python
RegionReported(r, s) :- PeakReported(x, y, z, s) & RegionVoxel(r, x, y, z)
```

which translates, in plain English, to 'region *r* is reported by study *s* if *s* reports a peak at location 
(*x, y, z*) that falls within region *r*'.

We model the reporting of networks by studies *probabilistically*, based on the reported regions that belong to each 
network, to account for the uncertainty in the location of reported peak activation coordinates. More precisely, 
each study has a probability of being considered to be reporting a network, equal to the reported volumetric proportion 
of the network in the study.

In plain English, 'a network *n* is considered to be reported by study *s* with probability $ \frac{v}{V} $ , where *v* is the 
sum of the volumes of regions within network *n* that are reported by study *s*, and *V* is the total volume of the 
network'. This is implemented by the following deterministic and probabilistic rules :

```python
RegionVolume(r, count(x, y, z) * resolution) :- RegionVoxel(r, x, y, z) & resolution = 3
NetworkVolume(n, sum(v)) :- RegionVolume(r, v) & NetworkRegion(n, r)
ReportedVolume(n, s, sum(v)) :- RegionVolume(r, v) & NetworkRegion(n, r) & RegionReported(r, s)
NetworkReported(n, s) : v/V :- ReportedVolume(n, s, v) & NetworkVolume(n, V)
```

These rules make use of NeuroLang’s built-in **count** and **sum** aggregation functions.

Finally, once we have a definition for whether a network is reported by a study, we define a rule that 
infers the probability that studies are associated with each topic given that they report only one of the 
three networks, as follows :

```python
ans(t, n, PROB) :- TopicAssociation(t, s) & SelectedStudy(s) // NetworkReported(n, s) & ~exists(n2; Network(n2) & n2 != n & NetworkReported(n2, s))
```

where the **//** operator represents a probabilistic conditioning.
"""


# %%
query = r"""tfidf :: TermInStudy(t, s) :- NeuroQueryTFIDF(tfidf, t, s)
TopicAssociation(topic, s) :- TermInStudy(term, s) & TopicTerm(topic, term)
RegionReported(r, s) :- PeakReported(i, j, k, s) & RegionVoxel(r, i, j, k)
RegionVolume(r, agg_count(i, j, k)) = RegionVoxel(r, i, j, k)
NetworkVolume(n, agg_sum(v)) :- RegionVolume(r, v) & NetworkRegion(n, r)
NetworkReportedVolume(network, study, agg_sum(volume)) :- NetworkRegion(network, region) & RegionReported(region, study) & RegionVolume(region, volume)
prob :: NetworkReported(network, study) :- NetworkVolume(network, nv) & NetworkReportedVolume(network, study, nrv) & (prob == nrv / nv)
ProbTopicAssociationGivenNetworkActivation(t, n, PROB(t, n)) :- (TopicAssociation(t, s) & SelectedStudy(s)) // (NetworkReported(n, s) & SelectedStudy(s))
ProbTopicAssociationGivenNoNetworkActivation(t, n, PROB(t, n)) :- (TopicAssociation(t, s) & SelectedStudy(s)) // (~NetworkReported(n, s) & Network(n) & SelectedStudy(s))
ProbTopicAssociation(t, PROB(t)) :- TopicAssociation(t, s) & SelectedStudy(s)
CountStudies(agg_count(s)) :- Study(s)
CountStudiesTopicAssociation(t, scount) :- ProbTopicAssociation(t, prob) & CountStudies(N) & (scount == prob * N)
ProbNetworkReported(n, PROB(n)) :- NetworkReported(n, s) & SelectedStudy(s)
CountStudiesNetworkReported(n, scount) :- ProbNetworkReported(n, prob) & CountStudies(N) & (scount == prob * N)
ProbTopicAssociationAndNetworkReported(t, n, PROB(t, n)) :- TopicAssociation(t, s) & NetworkReported(n, s) & SelectedStudy(s)
CountStudiesTopicAssociationAndNetworkReported(t, n, scont) :- ProbTopicAssociationAndNetworkReported(t, n, prob) & CountStudies(N) & (scont == prob * N)
Counts(topic, network, N, n, m, k) :- CountStudies(N) & CountStudiesTopicAssociation(topic, m) & CountStudiesNetworkReported(network, n) & CountStudiesTopicAssociationAndNetworkReported(topic, network, k)
Query(topic, network, p_topic, p_network, p0, p1, llr, N, n, m, k) :- ProbTopicAssociation(topic, p_topic) & ProbNetworkReported(network, p_network) & ProbTopicAssociationGivenNoNetworkActivation(topic, network, p0) & ProbTopicAssociationGivenNetworkActivation(topic, network, p1) & Counts(topic, network, N, n, m, k) & ( llr == ( k * log(p1) + ((n - k) * log(1 - p1) + ((m - k) * log(p0) + (((N - n) - (m - k)) * log(1 - p0))))) - ( k * log(p_topic) + ((n - k) * log(1 - p_topic) + ((m - k) * log(p_topic) + (((N - n) - (m - k)) * log(1 - p_topic))))))
"""

# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, query)
qw

# %% [markdown]
"""
We observe that topics related to sensory processing of direct environmental demands such as eye movements, visual attention, action, 
and spatial location are more likely to appear in studies reporting activations in the DAN than those reporting activation in the FPCN 
or DMN. We also observe that topics related to domain-general cognitive functions such as decision making, task switching, task demands, 
response inhibition, and performance monitoring are more likely to be mentioned in studies reporting activations in the FPCN than in the 
DAN and DMN. Finally, topics related to higher-order abstract cognitive and memory-related processes are mostly associated with studies 
reporting DMN activations than those reporting FPCN or DAN.
"""

# %% [markdown]
# ### References
# <a id="1">[1]</a>
# R. Nathan Spreng, Jorge Sepulcre, Gary R. Turner, W. Dale Stevens, and Daniel L. Schacter. 
# Intrinsic architecture underlying the relations among the default, dorsal attention, and frontoparietal 
# control networks of the human brain. *Journal of Cognitive Neuroscience*, 25(1):74–86, 
# January 2013. ISSN 1530-8898. doi: 10.1162/jocn a 00281.
#
# <a id="2">[2]</a>
# Matthew L. Dixon, Alejandro De La Vega, Caitlin Mills, Jessica Andrews-Hanna, R. Nathan Spreng, Michael W. Cole, and Kalina Christoff. 
# Heterogeneity within the frontoparietal control network and its relationship to the default and dorsal attention networks. 
# *Proceedings of the National Academy of Sciences*, 115(7):E1598, February 2018. doi: 10.1073/pnas.1715766115. 
# URL http://www.pnas.org/content/115/7/E1598.abstract.
#
# <a id="3">[3]</a>
# Radek Ptak, Armin Schnider, and Julia Fellrath. The Dorsal Frontoparietal Network: A Core System for Emulated Action. 
# *Trends in Cognitive Sciences*, 21(8):589–599, August 2017. ISSN 1879-307X. doi: 10.1016/j.tics.2017.05.002.

# %%
