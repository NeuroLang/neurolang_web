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

# %%
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Callable

import nibabel as nib
import numpy as np
from neurolang.frontend import NeurolangPDL

from gallery import metafc

# %%
data_dir = Path("neurolang_data")

# %%
def init_frontend():
    nl = NeurolangPDL()

    nl.add_symbol(
        np.log, name="log", type_=Callable[[float], float],
    )

    return nl


# %%
def load_studies(
    nl,
    term_association,
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
    term_association = term_association.loc[
        term_association.study_id.isin(study_ids.study_id)
    ]
    nl.add_probabilistic_facts_from_tuples(
        set(
            term_association[["tfidf", "term", "study_id"]].itertuples(
                index=False, name=None
            )
        ),
        name="TermAssociation",
    )
    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_uniform_probabilistic_choice_over_set(
        study_ids, name="SelectedStudy"
    )
    nl.add_tuple_set(study_ids, name="Study")


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
mni_mask = metafc.load_mni_atlas(data_dir, resolution=resolution)

# %%
coord_type = "xyz"
term_association, peak_reported, study_ids = metafc.load_neuroquery(
    data_dir, mni_mask, tfidf_threshold=None, coord_type=coord_type
)

# %%
nl = init_frontend()

# %%
load_studies(nl, term_association, peak_reported, study_ids)
load_attention_language_networks(nl)

# %%
query = r"""
RegionReported(r, s) :- PeakReported(x1, y1, z1, s) & RegionSeedVoxel(r, x2, y2, z2) & (d == EUCLIDEAN(x1, y1, z1, x2, y2, z2)) & (d < 10.0)
NetworkReported(n, s) :- RegionReported(r, s) & RegionInNetwork(r, n)
StudyMatchingNetworkQuery(s, n) :- RegionReported("VWFA", s) & NetworkReported(n, s) & exists(n2; ((n2 != n) & NetworkReported(n2, s) & Network(n2)))
StudyNotMatchingSegregationQuery(s, n) :- ~StudyMatchingNetworkQuery(s, n) & Study(s) & Network(n)
PositiveReverseInferenceSegregationQuery(t, n, PROB(t, n)) :- (TermAssociation(t, s) & SelectedStudy(s)) // (StudyMatchingNetworkQuery(s, n) & SelectedStudy(s))
NegativeReverseInferenceSegregationQuery(t, n, PROB(t, n)) :- (TermAssociation(t, s) & SelectedStudy(s)) // (StudyNotMatchingSegregationQuery(s, n) & SelectedStudy(s))
MarginalTermAssociation(t, PROB(t)) :- SelectedStudy(s) & TermAssociation(t, s)
CountStudies(count(s)) :- Study(s)
CountStudiesWithTerm(t, c) :- MarginalTermAssociation(t, prob) & (c == prob * N) & CountStudies(N)
CountStudiesMatchingQuery(n, count(s)) :- StudyMatchingNetworkQuery(s, n)
JointProb(t, n, PROB(t, n)) :- TermAssociation(t, s) & StudyMatchingNetworkQuery(s, n) & SelectedStudy(s)
CountStudiesMatchingQueryWithTerm(t, n, c) :- JointProb(t, n, p) & CountStudies(N) & (c == N * p)
LikelihoodRatio(term, network, p1, p0, llr, m, n, k) :- PositiveReverseInferenceSegregationQuery(term, network, p1) & NegativeReverseInferenceSegregationQuery(term, network, p0) & MarginalTermAssociation(term, p) & CountStudies(N) & CountStudiesWithTerm(term, m) & CountStudiesMatchingQuery(network, n) & CountStudiesMatchingQueryWithTerm(term, network, k) & ( llr == ( k * log(p1) + ((n - k) * log(1 - p1) + ((m - k) * log(p0) + (((N - n) - (m + k)) * log(1 - p0))))) - ( k * log(p) + ((n - k) * log(1 - p) + ((m - k) * log(p) + (((N - n) - (m + k)) * log(1 - p))))))
ans(term, network, pTgQ, pTgNotQ, llr, nb_studies_associated_with_term, nb_studies_matching_segregation_query, nb_studies_both_associated_with_term_and_matching_query) :- LikelihoodRatio(term, network, pTgQ, pTgNotQ, llr, nb_studies_associated_with_term, nb_studies_matching_segregation_query, nb_studies_both_associated_with_term_and_matching_query)
"""

# %%
with nl.scope:
    res = nl.execute_datalog_program(query)

res
# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, query)
qw

# %%
