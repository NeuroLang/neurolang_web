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
from nilearn import datasets, image, plotting
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
    def agg_create_region(i: Iterable, j: Iterable, k: Iterable) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBR(voxels, mni_mask.affine, image_dim=mni_mask.shape)

    # Aggregation function to build a region overlay
    @nl.add_symbol
    def agg_create_region_overlay(
        i: Iterable, j: Iterable, k: Iterable, p: Iterable
    ) -> ExplicitVBR:
        voxels = np.c_[i, j, k]
        return ExplicitVBROverlay(voxels, mni_mask.affine, p, image_dim=mni_mask.shape)

    return nl


# %% [markdown]
"""
### Fetch the data from atlases
We start by loading the data we need from available atlases.

We use the Neurosynth CBMA database (Yarkoni et al., 2011), consisting of 14,371 studies, 
its associated `v5-topics-200` topic model (Poldrack et al., 2012), and the DiFuMo functional atlas.
"""

# %%
# MNI Template
resolution = 3
mni_mask = data_utils.load_mni_atlas(data_dir=data_dir, resolution=resolution, key="t1")

# %%
# Difumo 128 Regions Atlas
coord_type = "ijk"
n_components = 128
region_voxels, difumo_meta = data_utils.fetch_difumo(
    data_dir=data_dir, mask=mni_mask, coord_type=coord_type, n_components=n_components,
)

difumo_img = datasets.fetch_atlas_difumo(
    dimension=n_components, data_dir=str(data_dir)
).maps

# %%
# NeuroSynth database
term_in_study, peak_reported, study_ids = data_utils.fetch_neurosynth(data_dir=data_dir)
ijk_positions = np.round(
    data_utils.xyz_to_ijk(peak_reported[["x", "y", "z"]].values.astype(float), mni_mask)
)
peak_reported["i"] = ijk_positions[:, 0]
peak_reported["j"] = ijk_positions[:, 1]
peak_reported["k"] = ijk_positions[:, 2]

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

We also generate random splits of the Neurosynth Dataset with 80% of studies in each split to compute confidence intervals.
We select a small number of splits to be able to compute the results in reasonable time, but this parameter can be increased.
The ids of the studies in each splits are stored in the **StudySplits** relation.
"""
# %%
from sklearn.model_selection import ShuffleSplit


def load_studies(
    nl,
    region_voxels,
    peak_reported,
    study_ids,
    n_splits,
    subsample_proportion: float = 0.8,
):
    nl.add_tuple_set(peak_reported, name="PeakReported")
    nl.add_tuple_set(region_voxels, name="RegionVoxel")
    nl.add_tuple_set(study_ids, name="Study")
    nl.add_uniform_probabilistic_choice_over_set(study_ids, name="SelectedStudy")
    splits = []
    for i, (train, _) in enumerate(
        ShuffleSplit(n_splits=n_splits, train_size=subsample_proportion).split(
            study_ids
        )
    ):
        split = study_ids.iloc[train].copy()
        split["split"] = i
        splits.append(split)
    splits = pd.concat(splits)
    nl.add_tuple_set(splits, name="StudySplits")


# %%
n_splits = 5
load_studies(nl, region_voxels, peak_reported, study_ids, n_splits=n_splits)

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
        n_topics, data_dir=data_dir, topics_to_keep=topics_to_keep, labels=labels,
    )

    nl.add_probabilistic_facts_from_tuples(
        topic_association[["prob", "topic", "study_id"]].itertuples(index=False),
        name="TopicInStudy",
    )


# %%
load_topics(nl)

# %% [markdown]
"""
### Formulate a NeuroLang query that extracts the DiFuMo regions in the PCC

Using neurolang's formalism, we can select the regions in the DiFuMo atlas which are in the PCC by looking at their labels: 

```python
PCCRegion(difumo_label, i, j, k) :- RegionVoxel(difumo_label, i, j, k) & startswith("Posterior cingulate", difumo_label)
```

This query reads: A region is in PCC if it has voxels (**RegionVoxel**) in the DiFuMo atlas 
and its label starts (**startswith**) with Posterior cingulate.
"""

# %%
pcc_query = r"""
PCCRegion(difumo_label, i, j, k) :- RegionVoxel(difumo_label, i, j, k) & startswith("Posterior cingulate", difumo_label)
ans(difumo_label, i, j, k) :- PCCRegion(difumo_label, i, j, k)
"""

# %%
pcc_region = nl.execute_datalog_program(pcc_query).as_pandas_dataframe()

# %% [markdown]
"""
### Categorize the PCC regions into Dorsal (dPCC) and Ventral (vPCC)

Once the PCC regions are identified, we categorize them into either Dorsal and Ventral, which allows us to define more
relations in Neurolang:

* **PCClabel** is a relation, or tuple, that stores the labels dPCC or vPCC.
* **RegionLabel**  is a relation, or tuple, that stores the labels of DiFuMo PCC regions.
* **RegionOfInterest** is a relation, or tuple, that stores the label of each DiFuMo region within the PCC, its voxel coordinates (i, j, k) and a column describing its position in the PCC (dPCC vs. vPCC).
"""

# %%
dpcc = np.where(
    (pcc_region["difumo_label"].values == "Posterior cingulate cortex antero-inferior")
    | (pcc_region["difumo_label"].values == "Posterior cingulate cortex superior")
)

vpcc = np.where(
    (pcc_region["difumo_label"].values == "Posterior cingulate cortex")
    | (pcc_region["difumo_label"].values == "Posterior cingulate cortex inferior")
)

pcc_region["pcc_label"] = ""
pcc_region["pcc_label"].iloc[dpcc] = "dPCC"
pcc_region["pcc_label"].iloc[vpcc] = "vPCC"

# %%
def add_pcc_regions(nl, pcc_region):
    nl.add_tuple_set([("dPCC",), ("vPCC",)], name="PCCLabel")
    nl.add_tuple_set(np.unique(pcc_region["difumo_label"].values), name="RegionLabel")
    nl.add_tuple_set(pcc_region, name="RegionOfInterest")


# %%
add_pcc_regions(nl, pcc_region)

# %% [markdown]
"""
### Visualize the Regions of Interest

Using an aggregate function, we can visualize the regions in the PCC. We create a **RegionOfInterestImage** relation which groups the voxels in the **RegionOfInterest** relation by *difumo_label* and *pcc_label* and creates an overlay image.

---
**NOTE**

Select the **RegionOfInterestImage** symbol in the result tab of the widget below to see the regions on the brain map.
The other tabs show all the relations which have been added to the Neurolang engine so far.

---
"""

# %%
agg_query = r"""
RegionOfInterestImage(difumo_label, pcc_label, agg_create_region(i, j, k)) :- RegionOfInterest(difumo_label, i, j, k, pcc_label)
"""

# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(nl, agg_query)
qw

# %% [markdown]
"""
### Formulate a NeuroLang Program

We're now able to create a neurolang program that derives selective functionnal profiles for each of the Dorsal and Ventral
PCC using a segregation query. For that we write a datalog program which consists of several rules. 

1. The first one finds the studies that report activation in the PCC:
```python
RegionReported(r, s) :- PeakReported(x1, y1, z1, s, i, j, k) & RegionOfInterest(difumo_label, i, j, k, r)
```

2. The second rule computes the marginal probability of finding a topic in a study in each of the splits of the database.
```python
MarginalProbTopicInStudy(t, split, PROB(t, split)) :- TopicInStudy(t, s)  & SelectedStudy(s)  & StudySplits(s, split)
```

3. We then write a segregation query that selects studies reporting activation in one PCC sub-region and not 
the other sub-region:
```python
StudyMatchingRegionSegregationQuery(s, r) :-  RegionReported(r, s) & ~RegionReported(r2, s) & RegionLabel(r2) & (r2 != r)
```

4. Which allows us to compute the probability of a topic being present given a segregation query:
```python
ProbTopicGivenRegionSegregationQuery(t, r, split, PROB(t, r, split)) :- (TopicInStudy(t, s)) // ( StudyMatchingRegionSegregationQuery(s, r) & SelectedStudy(s) & StudySplits(s, split) )    
```

5. Finally, we combine the above queries to calculate the log-odds ratio of topic associations for the dPCC and vPCC:
```python
TopicRegionAssociationLogOdds(t, r, split, p, pmarg, logodds) :- ProbTopicGivenRegionSegregationQuery(t, r, split, p)  & MarginalProbTopicInStudy(t, split, pmarg)  & (logodds == log_odds(p, pmarg))
```
"""

# %%
query = r"""
RegionReported(r, s) :- PeakReported(x1, y1, z1, s, i, j, k) & RegionOfInterest(difumo_label, i, j, k, r)
MarginalProbTopicInStudy(t, split, PROB(t, split)) :- TopicInStudy(t, s)  & SelectedStudy(s)  & StudySplits(s, split)
StudyMatchingRegionSegregationQuery(s, r) :-  RegionReported(r, s) & ~RegionReported(r2, s) & RegionLabel(r2) & (r2 != r)
ProbTopicGivenRegionSegregationQuery(t, r, split, PROB(t, r, split)) :- (TopicInStudy(t, s)) // ( StudyMatchingRegionSegregationQuery(s, r) & SelectedStudy(s) & StudySplits(s, split) )    
TopicRegionAssociationLogOdds(t, r, split, p, pmarg, logodds) :- ProbTopicGivenRegionSegregationQuery(t, r, split, p)  & MarginalProbTopicInStudy(t, split, pmarg)  & (logodds == log_odds(p, pmarg))
ans(t, r, split, p, pmarg, logodds) :-  TopicRegionAssociationLogOdds(t, r, split, p, pmarg, logodds)
"""

# %% [markdown]
"""
### Define utilitary plot functions

Before running the program, we define some utility functions to plot the results. The `make_radar_plot` function
will create a plot displaying the logodds probability for each region of being associated with one of the networks.
The `make_histogram_plot` function displays a histogram plot of log-odds ratios across the splits. 
Both functions are passed a dictionnary containing the result sets of the query that is executed by Neurolang.
"""

# %%
# %matplotlib widget
import seaborn as sns
import matplotlib.pyplot as plt


def make_radar_plot(results):
    """Create a radar plot using the query results

    Parameters
    ----------
    results : Dict[str, NamedRelationalAlgebraFrozenSet]
        a dictionary of name -> relational algebra result sets which contains all
        the relations returned by the Neurolang program
    """
    solution = results["ans"]
    d = solution.as_pandas_dataframe()

    topics = list(sorted(d["t"].unique()))
    network_to_label = {
        "dPCC": "dorsal PCC",
        "vPCC": "ventral PCC",
    }

    n_topics = len(topics)
    angles = np.linspace(0, 2 * np.pi, n_topics, endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(4.5, 3), subplot_kw=dict(polar=True), dpi=200)
    fig.canvas.header_visible = False
    colors = {"dPCC": "black", "vPCC": "purple"}
    for network, dn in d.groupby("r"):
        dn = dn.groupby("t").agg({"logodds": ["mean", "std"]}).loc[topics]
        p_mean = list(dn[("logodds", "mean")])
        p_std = list(dn[("logodds", "std")])
        p_mean += p_mean[:1]
        p_std += p_std[:1]
        ax.plot(
            angles,
            p_mean,
            linewidth=1,
            label=network_to_label[network],
            color=colors[network],
            marker="o",
            markersize=4,
        )
        ax.fill_between(
            angles,
            np.array(p_mean) - np.array(p_std),
            np.array(p_mean) + np.array(p_std),
            color=colors[network],
            alpha=0.4,
        )

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(
        np.degrees(angles[:-1]), topics, fontsize=5,
    )
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, np.pi):
            label.set_horizontalalignment("center")
        elif 0 < angle < np.pi:
            label.set_horizontalalignment("left")
        else:
            label.set_horizontalalignment("right")
    ax.set_rlabel_position(180 / n_topics)
    ax.legend(
        bbox_to_anchor=(-0.4, 0.98), loc=3, fontsize=5,
    )
    ax.spines["bottom"] = ax.spines["inner"]


# %%
def make_histogram_plot(results):
    """Create a histogram plot using the query results

    Parameters
    ----------
    results : Dict[str, NamedRelationalAlgebraFrozenSet]
        a dictionary of name -> relational algebra result sets which contains all
        the relations returned by the Neurolang program
    """
    solution = results["ans"]
    d = solution.as_pandas_dataframe()
    solution_report = (
        d.groupby(["t", "r"])["p"]
        .aggregate(
            [
                "mean",
                lambda x: np.mean(x) - 1.96 * np.std(x) / np.sqrt(len(x)),
                lambda x: np.mean(x) + 1.96 * np.std(x) / np.sqrt(len(x)),
                "std",
            ]
        )
        .rename(columns={"<lambda_0>": "C.I. 95% -", "<lambda_1>": "C.I. 95% +"})
    )

    pooled_std = np.sqrt(
        (
            solution_report.swaplevel(i=0, j=1).loc["dPCC", "std"] ** 2
            + solution_report.swaplevel(i=0, j=1).loc["vPCC", "std"] ** 2
        )
        / 2
    )

    z_difference = (
        solution_report.swaplevel(i=0, j=1).loc["dPCC", "mean"]
        - solution_report.swaplevel(i=0, j=1).loc["vPCC", "mean"]
    ) / pooled_std

    z_difference_sorted_difference = (
        np.abs(z_difference).sort_values(ascending=False).index
    )

    d = d.rename(
        columns={
            "t": "Cognitive Function",
            "r": "PCC Sub-Region",
            "logodds": "Log-Odds",
        }
    )
    fg = sns.FacetGrid(
        d,
        col="Cognitive Function",
        col_wrap=4,
        hue="PCC Sub-Region",
        col_order=z_difference_sorted_difference,
    )
    fg.fig.canvas.header_visible = False
    fg.map(sns.kdeplot, "Log-Odds", shade=True)
    fg.add_legend()
    plt.setp(fg._legend.get_title(), fontsize=15)
    fg.fig.suptitle(
        f"Functional Segregation Query of dPCC and vPCC Sorted by Approximate Evidence Amount"
        " of Segregation\n"
        f"(Histograms across {n_splits:,} Splits with 80% "
        f"of {len(study_ids):,} Articles from the Neurosynth CBMA Database)",
        y=1.08,
        size=18,
    )
    fg.set_titles(col_template="{col_name}")


# %%
from nlweb.viewers.query import QueryWidget

qw = QueryWidget(
    nl,
    query,
    callbacks={
        "Radar Plot": make_radar_plot,
        "Log-Odds Histogram": make_histogram_plot,
    },
)
qw


# %%
