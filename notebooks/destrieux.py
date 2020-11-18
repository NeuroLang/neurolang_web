# -*- coding: utf-8 -*-
# ## Destrieux cortical atlas (dated 2009)

# +
import sys
import os
import warnings  # type: ignore

warnings.filterwarnings("ignore")

from neurolang import regions  # type: ignore
from neurolang.frontend import NeurolangDL, ExplicitVBR  # type: ignore

from nlweb.viewers.query import QueryWidget

import nibabel as nib

from nilearn import datasets  # type: ignore

# Query agent
def init_agent():
    """
    Set up the neurolang query runner (?) and add facts (?) to
    the database
    """
    nl = NeurolangDL()

    @nl.add_symbol
    def region_union(rs):
        return regions.region_union(rs)

    return nl


def add_destrieux(nl):
    nl.new_symbol(name="destrieux")
    destrieux_atlas = datasets.fetch_atlas_destrieux_2009(data_dir="neurolang_data")
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


with open(os.devnull, "w") as devnull:
    old_stdout = sys.stdout
    sys.stdout = devnull

    # Prepare engine
    nl = init_agent()
    add_destrieux(nl)

    sys.stdout = old_stdout

# display query gui
query = "ans(region_union(r)) :- destrieux(..., r)"
qw = QueryWidget(nl, query)
qw
