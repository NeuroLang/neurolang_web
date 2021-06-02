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
# # Coactivation Analysis of the Functional Connectivity Differences Between Two FPCN Subnetworks

# %% [markdown]
"""
A running hypothesis is that the frontoparietal cognitive control network (FPCN) can be decomposed into subsystems 
associated with disparate and overlapping mental processes. Dixon et al. [[1]](#1) studied two broad subsystems of the FPCN 
that also appear as separate networks in the influential 17-network model from Yeo et al. [[2]](#2). 
Similarly to Dixon et al. [[1]](#1), we name these two subsystems FPCN-A and FPCN-B. Dixon et al. [[1]](#1) observed significant 
couplings between FPCN-A and the default mode network (DMN), and between FPCN-B and the dorsal attention network (DAN). 
We replicate theses results by conducting a similar meta-analysis with NeuroLang.
"""

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable, Iterable

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
    nl.add_symbol(
        lambda it: float(sum(it)),
        name="agg_sum",
        type_=Callable[[Iterable], float],
    )

    @nl.add_symbol
    def agg_count(*iterables) -> int:
        return len(next(iter(iterables)))

    return nl


# %%
def load_studies(
    nl,
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

    nl.add_tuple_set(peak_reported, name="PeakReported")
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
_, peak_reported, study_ids = data_utils.fetch_neuroquery(
    mask=mni_mask, coord_type=coord_type, data_dir=data_dir
)

# %%
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
load_studies(nl, peak_reported, study_ids)
load_voxels(nl, region_voxels, difumo_meta)

# %% [markdown]
"""
Our approach is to formulate conditional probabilistic queries that include studies reporting activations in each 
of the two FPCN subnetworks. By contrasting their probabilistic maps, we identify a distinct coactivation pattern 
associated with each subnetwork. 

We model the reporting of networks by studies *probabilistically*, based on the reported regions that belong to each 
network, to account for the uncertainty in the location of reported peak activation coordinates. More precisely, 
each study has a probability of being considered to be reporting a network, equal to the reported volumetric proportion 
of the network in the study.

We then formulate a rule that calculates the coactivation pattern of each FPCN subnetwork. In NeuroLang we use the 
following rule to calculate the conditional probability of a region being reported given that a network is also reported:

```python
ans(r, n, PROB) :- RegionReported(r, s) & SelectedStudy(s) // NetworkReported(n, s) & SelectedStudy(s)
```

whose resulting **ans** table contains tuples (*r, n, p*), where *p* is the probability of region *r* being reported by 
studies reporting network *n*, *n* being either FPCN-A or FPCN-B. We use a likelihood-ratio test, and a FDR correction 
for multiple comparison, to identify significant coactivating regions.
"""


# %%
query = r"""RegionReported(r, s) :- PeakReported(i, j, k, s) & RegionVoxel(r, i, j, k)
RegionVolume(r, agg_count(i, j, k)) :- RegionVoxel(r, i, j, k)
NetworkVolume(n, agg_sum(v)) :- RegionVolume(r, v) & NetworkRegion(n, r)
NetworkReportedVolume(network, study, agg_sum(volume)) :- NetworkRegion(network, region) & RegionReported(region, study) & RegionVolume(region, volume)
prob :: NetworkReported(network, study) :- NetworkVolume(network, nv) & NetworkReportedVolume(network, study, nrv) & (prob == nrv / nv)
ProbActivationGivenNetworkActivation(r, n, PROB(r, n)) :- (RegionReported(r, s) & SelectedStudy(s)) // (NetworkReported(n, s) & SelectedStudy(s))
ProbActivationGivenNoNetworkActivation(r, n, PROB(r, n)) :- (RegionReported(r, s) & SelectedStudy(s)) // (~NetworkReported(n, s) & Network(n) & SelectedStudy(s))
ProbActivation(r, PROB(r)) :- RegionReported(r, s) & SelectedStudy(s)
CountStudies(agg_count(s)) :- Study(s)
CountStudiesRegionReported(r, agg_count(s)) :- RegionReported(r, s)
ProbNetworkReported(n, PROB(n)) :- NetworkReported(n, s) & SelectedStudy(s)
CountStudiesNetworkReported(n, scount) :- ProbNetworkReported(n, prob) & CountStudies(N) & (scount == prob * N)
ProbRegionAndNetworkReported(r, n, PROB(r, n)) :- RegionReported(r, s) & NetworkReported(n, s) & SelectedStudy(s)
CountStudiesRegionAndNetworkReported(r, n, scount) :- ProbRegionAndNetworkReported(r, n, prob) & CountStudies(N) & (scount == prob * N)
Counts(region, network, N, n, m, k) :- CountStudies(N) & CountStudiesRegionReported(region, m) & CountStudiesNetworkReported(network, n) & CountStudiesRegionAndNetworkReported(region, network, k)
Query(region, network, p, p0, p1, llr, N, n, m, k) :- ProbActivation(region, p) & ProbActivationGivenNoNetworkActivation(region, network, p0) & ProbActivationGivenNetworkActivation(region, network, p1) & Counts(region, network, N, n, m, k) & ( llr == ( k * log(p1) + ((n - k) * log(1 - p1) + ((m - k) * log(p0) + (((N - n) - (m - k)) * log(1 - p0))))) - ( k * log(p) + ((n - k) * log(1 - p) + ((m - k) * log(p) + (((N - n) - (m - k)) * log(1 - p))))))
ans(region, network, p, p0, p1, llr, N, n, m, k) :- Query(region, network, p, p0, p1, llr, N, n, m, k)"""

# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, query)
qw

# %% [markdown]
"""
### References
<a id="1">[1]</a>
Matthew L. Dixon, Alejandro De La Vega, Caitlin Mills, Jessica Andrews-Hanna, R. Nathan Spreng, Michael W. Cole, and Kalina Christoff. 
Heterogeneity within the frontoparietal control network and its relationship to the default and dorsal attention networks. 
*Proceedings of the National Academy of Sciences*, 115(7):E1598, February 2018. doi: 10.1073/pnas.1715766115. 
URL http://www.pnas.org/content/115/7/E1598.abstract.

<a id="2">[2]</a>
BT Thomas Yeo, Fenna M Krienen, Jorge Sepulcre, Mert R Sabuncu, Danial Lashkari, Marisa Hollinshead, Joshua L Roffman, 
Jordan W Smoller, Lilla Zollei, Jonathan R Polimeni, and others. The organization of the human cerebral cortex estimated 
by intrinsic functional connectivity. *Journal of neurophysiology, 2011*. Publisher: American Physiological Society Bethesda, MD.
"""

# %%
