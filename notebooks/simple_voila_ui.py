# ## Neurolang Query UI

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


# ## UI utilities


class PapayaWidget(HTML):
    def __init__(self, *args, **kwargs):
        super(PapayaWidget, self).__init__(*args, **kwargs)

        self.params = {"kioskMode": False, "worldSpace": True, "fullScreen": False}

        self.atlas_image = nib.load("avg152T1_brain.nii.gz")

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

    def _encode_images(self, images):
        encoded_images = []
        image_txt = []
        self.spatial_images = [self.atlas_image] + images
        for i, image in enumerate(self.spatial_images):
            encoded_image = base64.encodebytes(
                nib.Nifti2Image(image.get_fdata(), affine=image.affine).to_bytes()
            )
            image_txt.append(f"image{i}")
            enc = encoded_image.decode("utf8").replace("\n", "")
            encoded_images.append(f'var {image_txt[-1]}="{enc}";')

        encoded_images = "\n".join(encoded_images)
        return encoded_images, image_txt

    def plot(self, images):
        self.reset()
        if len(images) > 0:
            params = dict()
            params.update(self.params)

            encoded_images, image_names = self._encode_images(images)
            params["encodedImages"] = image_names

            for image_name in image_names[1:]:
                params[image_name] = {"min": 0, "max": 10, "lut": "Red Overlay"}

            if len(self.spatial_images) > 0:
                coords = (
                    np.transpose(self.spatial_images[-1].get_fdata().nonzero())
                    .mean(0)
                    .astype(int)
                )
                coords = nib.affines.apply_affine(
                    self.spatial_images[-1].affine, coords
                )
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


class PlotWidget(Output):
    def __init__(self, *args, **kwargs):
        super(PlotWidget, self).__init__(*args, **kwargs)

        self.display = None

    def plot(self, images):
        self.reset()
        if len(images) > 0:
            with self:
                image = next(iter(images))
                self.display = plotting.plot_roi(image)
                plotting.show()

    def reset(self):
        if self.display is not None:
            self.display.close()
        self.clear_output()


class TableSetWidget(VBox):
    selection = traitlets.Set()  # selected images in table

    def __init__(self, name: str, wras: WrappedRelationalAlgebraSet):
        super(TableSetWidget, self).__init__()

        self.wras = wras
        self.checkboxes = []

        name_label = HTML(f"<h2>{name}</h2>")
        select_all = Button(description="select all")
        select_all.on_click(self.select_all)
        clear_selection = Button(description="clear selection")
        clear_selection.on_click(self.unselect_all)

        header = HBox(
            [name_label, select_all, clear_selection],
            layout=Layout(align_items="center"),
        )
        self.sheet = self._init_sheet(self.wras, self.selection)

        self.children = [header, self.sheet]

    def _init_sheet(self, wras, selection):
        column_headers = [str(i) for i in range(wras.arity)]
        rows_visible = min(len(wras), 5)
        table = sheet(
            rows=len(wras),
            columns=wras.arity,
            column_headers=column_headers,
            layout=Layout(width="auto", height=f"{50 * rows_visible}px"),
        )

        def selection_changed(change, image):
            if change["new"]:
                self.selection = self.selection | {image}
            else:
                self.selection = self.selection - {image}

        for i, tuple_ in enumerate(wras.unwrapped_iter()):
            row_temp = []
            for el in tuple_:
                if isinstance(el, ExplicitVBR):
                    checkbox = Checkbox(value=False, description="show region")
                    checkbox.observe(
                        partial(selection_changed, image=el.spatial_image()),
                        names="value",
                    )
                    row_temp.append(checkbox)
                    self.checkboxes.append(checkbox)
                else:
                    row_temp.append(Label(str(el)))
            row(i, row_temp)
        return table

    def select_all(self, args, **kwargs):
        for cb in self.checkboxes:
            cb.value = True

    def unselect_all(self, args, **kwargs):
        for cb in self.checkboxes:
            cb.value = False


class ResultWidget(VBox):
    selection = traitlets.Set()  # union of selected images for each table in results

    def __init__(self):
        super(ResultWidget, self).__init__()

        self.viewer = PlotWidget(
            layout=Layout(width="500px", height="250px", border="1px solid black")
        )

        #self.viewer = PapayaWidget(layout = Layout(width='700px', height='600px', border='1px solid black'))

    def show_results(self, res: Dict[str, WrappedRelationalAlgebraSet]):
        self.selection = set()
        tablesets = self._create_tablesets(res)

        @debounce(0.2)
        def selection_changed(_):
            self.viewer.plot(self.selection)

        self.observe(selection_changed, names="selection")
        self.children = [self.viewer] + tablesets

    def _create_tablesets(self, res):
        tablesets = []
        for name, result_set in res.items():
            tableset_widget = TableSetWidget(name, result_set)

            def selection_changed(change):
                old = change["old"]
                new = change["new"]
                self.selection = (self.selection - (old - new)) | (new - old)

            tableset_widget.observe(selection_changed, names="selection")
            tablesets.append(tableset_widget)
        return tablesets

    def reset(self):
        self.viewer.reset()

        for table in self.children[1:]:
            table.close()

        self.children = [self.viewer]


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


# ## Query the NeuroLang engine and display results

# +
nl = init_agent()
default_query = "ans(region_union(r)) :- destrieux(..., r)"

qw = QueryWidget(nl, default_query)
qw
# -


