# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Destrieux cortical atlas (dated 2009)

# +
import sys
import os
import warnings  # type: ignore

warnings.filterwarnings("ignore")

import nibabel as nib
from nilearn import datasets  # type: ignore

from neurolang import regions  # type: ignore
from neurolang.frontend import NeurolangDL, ExplicitVBR  # type: ignore

from nlweb.viewers.query import QueryWidget


# +
def init_agent():
    nl = NeurolangDL()

    @nl.add_symbol
    def region_union(rs):
        return regions.region_union(rs)

    return nl


def add_destrieux(nl, destrieux_atlas):
    nl.new_symbol(name="destrieux")

    destrieux_atlas_image = nib.load(destrieux_atlas["maps"])
    destrieux_labels = dict(destrieux_atlas["labels"])

    destrieux_set = set()
    for k, v in destrieux_labels.items():
        if k == 0:
            continue
        destrieux_set.add(
            (
                v.decode("utf8"),
                ExplicitVBR.from_spatial_image_label(destrieux_atlas_image, k),
            )
        )

    nl.add_tuple_set(destrieux_set, name="destrieux")


def load_database():
    return datasets.fetch_atlas_destrieux_2009(data_dir="neurolang_data")


# +
# Prepare engine

# prevent stdout to ui in the gallery
with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull

    # Initialize query agent
    nl = init_agent()
    destrieux_atlas = load_database()
    add_destrieux(nl, destrieux_atlas)

    sys.stdout = old_stdout
# -

# Display query gui
query = "union(region_union(r)) :- destrieux(..., r)"
qw = QueryWidget(nl, query)
qw


