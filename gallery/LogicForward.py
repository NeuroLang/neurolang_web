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
# # Logical Topic Segregation Queries can Derive Differential Meta-Analytic Activation Patterns within the FPCN

# %% [markdown]
"""
In this example, we perform forward inference *topic-based segregation queries* to derive activation patterns within 
the frontoparietal cognitive control network (FPCN), given the presence of a psychological topic of interest, 
with an additional condition that studies included within the meta-analysis are *not associated* with all other topics 
of interest. Often referred to as the *multiple demand system* [[1]](#1), activity within the FPCN is associated with a large 
set of tasks, themselves belonging to disparate and overlapping cognitive components such as decision making, 
working memory, memory retrieval, task switching, and semantic processing, to name a few. Yet, the literature provides 
evidence of a heterogeneous internal organization, whereby a different combination of regions may be involved in a 
different domain of processing [[2]](#2). Thus, the main goal of this example is to identify unique activation patterns within 
the FPCN each predicted by the presence of a particular topic and the simultaneous absence of other topics.

**Segregation queries can maximise the specificity of meta-analytic forward inferences** by minimising the amount of 
overlap amongst nuanced topics. In this framework, the topic-based segregation query automatically selects the studies 
that predominantly load on a single topic representing a cognitive process known to be associated with FPCN activity, 
while simultaneously discarding studies that load on any of the other topics, which also represent cognitive processes 
associated with activity of the FPCN.
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
From a set of 200 topics, learned from a large corpus of studies using topic modeling and made available by Neurosynth [[3]](#3), 
we selected five topics representing a subset of the broad cognitive processes widely attributed to the FPCN, along 
with the loading values of studies on each topic: *working memory*, *decision making*, *task set switching*, 
*semantic processing*, and *memory retrieval*. This selection was based on the findings of Spreng et al. [[4]](#4), 
Duncan [[1]](#1), Yarkoni et al. [[5]](#5), Niendam et al. [[6]](#6).
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
    peak_reported,
    study_ids,
    split_id: int = 42,
    subsample_proportion: float = 0.6,
):
    n_studies_selected = int(len(study_ids) * subsample_proportion)
    np.random.seed(split_id)
    study_ids = study_ids.sample(n_studies_selected)
    peak_reported = peak_reported.loc[peak_reported.study_id.isin(study_ids.study_id)]
    topic_association = topic_association.loc[
        topic_association.study_id.isin(study_ids.study_id)
    ]
    topic_association = topic_association.loc[
        topic_association.prob > 0.05, ["topic", "study_id"]
    ]
    nl.add_tuple_set(
        topic_association, name="TopicAssociation",
    )
    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_uniform_probabilistic_choice_over_set(study_ids, name="SelectedStudy")
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_tuple_set(topic_association[["topic"]].drop_duplicates(), name="Topic")


# %%
def load_voxels(nl, region_voxels, mni_mask):
    nl.add_tuple_set(region_voxels, name="RegionVoxel")
    nl.add_tuple_set(
        nibabel.affines.apply_affine(
            mni_mask.affine, np.transpose(mni_mask.get_fdata().nonzero())
        ),
        name="Voxel",
    )


# %%
resolution = 3
mni_mask = data_utils.load_mni_atlas(data_dir=data_dir, resolution=resolution)


# %%
_, peak_reported, study_ids = data_utils.fetch_neurosynth(
    tfidf_threshold=1e-2, data_dir=data_dir
)

# %%
coord_type = "xyz"
n_difumo_components = 256
region_voxels, difumo_meta = data_utils.fetch_difumo(
    mask=mni_mask,
    coord_type=coord_type,
    n_components=n_difumo_components,
    data_dir=data_dir,
)


# %%
nl = init_frontend()

# %%
load_studies(nl, topic_association, peak_reported, study_ids)
load_voxels(nl, region_voxels, mni_mask)

# %% [markdown]
"""
Then, we formulate the following NeuroLang program which performs topic segregation queries, yielding an activation map 
for each topic separately:

```python
Match(t, s) :- TopicAssociation(t, s) & ~exists(t2; Topic(t2) & t2 != t & Study(s) & TopicAssociation(t2, s))
ans1(t, x, y, z, PROB) :- VoxelReported(x, y, z, s) & SelectedStudy(s) // Match(t, s) & SelectedStudy(s)
```
"""


# %%
query = r"""
VoxelReported(x, y, z, s) :- PeakReported(x2, y2, z2, s) & Voxel(x, y, z) & (d == EUCLIDEAN(x, y, z, x2, y2, z2)) & (d < 10)
Match(t, s) :- TopicAssociation(t, s) & ~exists(t2; (Topic(t2) & Study(s) & TopicAssociation(t2, s) & (t2 != t)))
NoMatch(t, s) :- ~Match(t, s) & Topic(t) & Study(s)
ans1(t, x, y, z, PROB(t, x, y, z)) :- (VoxelReported(x, y, z, s) & SelectedStudy(s)) // (Match(t, s) & SelectedStudy(s))
ans0(t, x, y, z, PROB(t, x, y, z)) :- (VoxelReported(x, y, z, s) & SelectedStudy(s)) // (NoMatch(t, s) & SelectedStudy(s))
Query(t, x, y, z, r, pAgMatch, pAgNoMatch) :- RegionVoxel(r, x, y, z) & ans1(t, x, y, z, pAgMatch) & ans0(t, x, y, z, pAgNoMatch)
ans(t, x, y, z, r, pAgMatch, pAgNoMatch) :- Query(t, x, y, z, r, pAgMatch, pAgNoMatch)
"""

# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, query)
qw

# %% [markdown]
"""
The results of this segregation query show that the FPCN exhibits a varied activation profile across topics, 
corroborating previous findings of flexible adaptation of activity within this network as task type change. 
Specifically, working memory and task set switching tend to activate, to some extent, spatially interleaved, 
frontal and parietal regions of the FPCN network.  Semantic processing, on the other hand, dominantly activates a 
left-lateralised frontal region anchored in the middle frontal gyrus and extends to the inferior frontal gyrus. 
Finally, decision making and memory retrieval are associated with activation in the cingulo-medial portion of the FPCN, 
the pre-supplementary motor/dorsal anterior cingulate cortex (decision making) and a precuneus/posterior cingulate 
cortex network (memory retrieval).

These results are in-line with existing findings from the literature and 
demonstrate the power of NeuroLang in expressing segregation queries capable of uncovering a network’s 
internal organisation.
"""

# %% [markdown]
"""
### References
<a id="1">[1]</a>
John Duncan. The multiple-demand (MD) system of the primate brain: mental programs for intelligent behaviour. 
*Trends in Cognitive Sciences*, 14(4):172–179, April 2010. ISSN 1364-6613. doi: 10.1016/j.tics.2010. 01.004. 
URL https://www.sciencedirect.com/science/article/pii/ S1364661310000057.

<a id="2">[2]</a>
Matthew L. Dixon, Alejandro De La Vega, Caitlin Mills, Jessica Andrews- Hanna, R. Nathan Spreng, Michael W. Cole, and Kalina Christoff. 
Heterogeneity within the frontoparietal control network and its relationship to the default and dorsal attention networks. 
*Proceedings of the National Academy of Sciences*, 115(7):E1598, February 2018. doi: 10.1073/pnas.1715766115. 
URL http://www.pnas.org/content/115/7/E1598.abstract.

<a id="3">[3]</a>
Russell A. Poldrack, Jeanette A. Mumford, Tom Schonberg, Donald Kalar, Bishal Barman, and Tal Yarkoni. 
Discovering Relations Between Mind, Brain, and Mental Disorders Using Topic Mapping. 
*PLoS Computational Biology*, 8(10):e1002707, October 2012. ISSN 1553-7358. doi: 10.1371/journal.pcbi.1002707. 
URL https://dx.plos.org/10.1371/ journal.pcbi.1002707.

<a id="4">[4]</a>
R. Nathan Spreng, Jorge Sepulcre, Gary R. Turner, W. Dale Stevens, and Daniel L. Schacter. 
Intrinsic architecture underlying the relations among the default, dorsal attention, and frontoparietal control networks 
of the human brain. *Journal of Cognitive Neuroscience*, 25(1):74–86, January 2013. ISSN 1530-8898. doi: 10.1162/jocn a 00281.

<a id="5">[5]</a>
Tal Yarkoni, Jeremy R. Gray, Elizabeth R. Chrastil, Deanna M. Barch, Leonard Green, and Todd S. Braver. 
Sustained neural activity associated with cognitive control during temporally extended decision making. 
*Brain Research. Cognitive Brain Research*, 23(1):71–84, April 2005. ISSN 0926- 6410. doi: 10.1016/j.cogbrainres.2005.01.013.

<a id="6">[6]</a>
Tara A. Niendam, Angela R. Laird, Kimberly L. Ray, Y. Monica Dean, David C. Glahn, and Cameron S. Carter. 
Meta-analytic evidence for a superordinate cognitive control network subserving diverse executive functions. 
*Cognitive, Affective & Behavioral Neuroscience*, 12(2):241–268, June 2012. ISSN 1531-135X. doi: 10.3758/s13415-011-0083-5.
"""
