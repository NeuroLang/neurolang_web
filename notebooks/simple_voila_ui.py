# ## Imports

import warnings  # type: ignore

warnings.filterwarnings("ignore")

# +
import base64  # type: ignore

from functools import partial

import html  # type: ignore

from ipysheet import row, sheet  # type: ignore
from ipywidgets import (
    Button,
    Checkbox,
    HBox,
    HTML,
    Label,
    Layout,
    Output,
    Textarea,
    VBox,
    Widget,
)  # type: ignore

import json  # type: ignore

from neurolang import regions  # type: ignore
from neurolang.datalog.wrapped_collections import (
    WrappedRelationalAlgebraSet,
)  # type: ignore
import neurolang
from neurolang.frontend import NeurolangDL, ExplicitVBR  # type: ignore

from nlweb.util import debounce

from nilearn import datasets, plotting  # type: ignore
import nibabel as nib  # type: ignore

import numpy as np  # type: ignore

import os  # type: ignore

import pandas as pd  # type: ignore

from typing import Dict

import traitlets


# -


# ## UI components

# ### Cell and cell viewer widgets


class LabelCellWidget(Label):
    def __init__(self, *args, **kwargs):
        super(LabelCellWidget, self).__init__(*args, **kwargs)

    def get_viewer(self):
        return None


class ExVBRCellWidget(Checkbox):
    def __init__(self, obj: neurolang.frontend.ExplicitVBR, *args, **kwargs):
        super(ExVBRCellWidget, self).__init__(*args, **kwargs)

        self.value = False
        self.description = "show region"

        def _selection_changed(change, image):
            if change["new"]:
                self.viewer.add(image)
            else:
                self.viewer.remove(image)

        self.observe(
            partial(_selection_changed, image=obj.spatial_image()), names="value"
        )

        self.viewer = ViewerFactory.get_region_viewer()

    def get_viewer(self):
        return self.viewer


class PapayaWidget(HTML):
    def __init__(self, *args, **kwargs):
        super(PapayaWidget, self).__init__(*args, **kwargs)

        self.params = {"kioskMode": False, "worldSpace": True, "fullScreen": False}

        self.atlas_image = nib.load("avg152T1_brain.nii.gz")

        self.spatial_images = [self.atlas_image]

        self.html = """
            <!DOCTYPE html>
            <html xmlns="http://www.w3.org/1999/xhtml" lang="en">
                <head>
                    <link rel="stylesheet" type="text/css" href="https://raw.githack.com/rii-mango/Papaya/master/release/current/standard/papaya.css" />
                    <script type="text/javascript" src="https://raw.githack.com/rii-mango/Papaya/master/release/current/standard/papaya.js"></script>
                    <title>Papaya Viewer</title>

                    <script type="text/javascript">

                        {encoded_images}

                        var params={params};
                    </script>
                </head>

                <body>
                    <div class="papaya" data-params="params"></div>
                </body>
            </html>
        """
        self.encoder = json.JSONEncoder()

    def _encode_images(self):
        encoded_images = []
        image_txt = []
        for i, image in enumerate(self.spatial_images):
            encoded_image = base64.encodebytes(
                nib.Nifti2Image(image.get_fdata(), affine=image.affine).to_bytes()
            )
            image_txt.append(f"image{i}")
            enc = encoded_image.decode("utf8").replace("\n", "")
            encoded_images.append(f'var {image_txt[-1]}="{enc}";')

        encoded_images = "\n".join(encoded_images)
        return encoded_images, image_txt

    def add(self, image):
        self.spatial_images.append(image)
        self.plot()

    def remove(self, image):
        self.spatial_images.remove(image)
        self.plot()

    def plot(self):
        self.reset()
        params = dict()
        params.update(self.params)

        encoded_images, image_names = self._encode_images()
        params["encodedImages"] = image_names

        for image_name in image_names[1:]:
            params[image_name] = {"min": 0, "max": 10, "lut": "Red Overlay"}

        if len(self.spatial_images) > 0:
            coords = (
                np.transpose(self.spatial_images[-1].get_fdata().nonzero())
                .mean(0)
                .astype(int)
            )
            coords = nib.affines.apply_affine(self.spatial_images[-1].affine, coords)
            params["coordinate"] = [int(c) for c in coords]

        escaped_papaya_html = html.escape(
            self.html.format(
                params=self.encoder.encode(params), encoded_images=encoded_images
            )
        )
        iframe = (
            f'<iframe srcdoc="{escaped_papaya_html}" id="papaya" '
            f'width="700px" height="600px"></iframe>'
        )
        self.value = iframe

    def reset(self):
        pass


# ###  Factories


class CellWidgetFactory:
    @staticmethod
    def get_cell_widget(obj):
        if isinstance(obj, neurolang.frontend.ExplicitVBR):
            return ExVBRCellWidget(obj)
        elif isinstance(obj, str) or isinstance(obj, neurolang.regions.EmptyRegion):
            return LabelCellWidget(str(obj))


class ViewerFactory:
    papaya_viewer = PapayaWidget(
        layout=Layout(width="700px", height="600px", border="1px solid black")
    )

    @staticmethod
    def get_region_viewer():
        return ViewerFactory.papaya_viewer


# ### Query and result widgets


class TableSetWidget(VBox):
    def __init__(self, name: str, wras: WrappedRelationalAlgebraSet, viewers: set):
        super(TableSetWidget, self).__init__()

        self.wras = wras
        self.cell_viewers = viewers

        # create widgets
        name_label = HTML(f"<h2>{name}</h2>")
        self.sheet = self._init_sheet(self.wras)

        self.children = [name_label, self.sheet]

    def _init_sheet(self, wras):
        column_headers = [str(i) for i in range(wras.arity)]
        rows_visible = min(len(wras), 5)
        table = sheet(
            rows=len(wras),
            columns=wras.arity,
            column_headers=column_headers,
            layout=Layout(width="auto", height=f"{50 * rows_visible}px"),
        )

        for i, tuple_ in enumerate(wras.unwrapped_iter()):
            row_temp = []
            for el in tuple_:
                cell_widget = CellWidgetFactory.get_cell_widget(el)
                row_temp.append(cell_widget)
                if cell_widget.get_viewer() is not None:
                    self.cell_viewers.add(cell_widget.get_viewer())
            row(i, row_temp)
        return table


class ResultWidget(VBox):
    def __init__(self):
        super(ResultWidget, self).__init__()

        self.viewers = set()

    def show_results(self, res: Dict[str, WrappedRelationalAlgebraSet]):
        tablesets = self._create_tablesets(res)

        self.children = tuple(self.viewers) + tuple(tablesets)

    def _create_tablesets(self, res):
        tablesets = []
        for name, result_set in res.items():
            tableset_widget = TableSetWidget(name, result_set, self.viewers)
            tablesets.append(tableset_widget)
        return tablesets

    def reset(self):
        for table in self.children[1:]:
            table.close()
        self.children = []


class QueryWidget(VBox):
    def __init__(self, neurolang_engine, default_query):
        super(QueryWidget, self).__init__()

        self.neurolang_engine = neurolang_engine

        self.query = Textarea(
            value=default_query,
            placeholder="Type something",
            disabled=False,
            layout=Layout(
                display="flex", flex_flow="row", align_items="stretch", width="75%"
            ),
        )
        self.button = Button(description="Run query")
        self.button.on_click(self._on_query_button_clicked)

        self.result_viewer = ResultWidget()

        self.children = [HBox([self.query, self.button]), self.result_viewer]

    def _on_query_button_clicked(self, b):
        self.result_viewer.reset()

        qresult = run_query(self.neurolang_engine, self.query.value)
        self.result_viewer.show_results(qresult)


# ## Query Agent


def init_agent():
    """
    Set up the neurolang query runner (?) and add facts (?) to
    the database
    """
    nl = NeurolangDL()

    @nl.add_symbol
    def region_union(rs):
        return regions.region_union(rs)

    destrieux_atlas = datasets.fetch_atlas_destrieux_2009()
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
    return nl


def run_query(nl, query):
    with nl.scope as s:
        nl.execute_nat_datalog_program(query)
        return nl.solve_all()


# ## Query the NeuroLang engine and display results

# +
nl = init_agent()
default_query = "ans(region_union(r)) :- destrieux(..., r)"

qw = QueryWidget(nl, default_query)
qw
# -
