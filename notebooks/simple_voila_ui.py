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
    DOMWidget,
    HBox,
    HTML,
    Label,
    Layout,
    Output,
    Textarea,
    VBox,
    Widget,
    register,
)  # type: ignore

import json  # type: ignore

from neurolang import regions  # type: ignore
from neurolang.datalog.wrapped_collections import (
    WrappedRelationalAlgebraSet,
)  # type: ignore
import neurolang
from neurolang.frontend import NeurolangDL, ExplicitVBR  # type: ignore
from neurolang.frontend.neurosynth_utils import StudyID, TfIDf

from nilearn import datasets, plotting  # type: ignore
import nibabel as nib  # type: ignore

import numpy as np  # type: ignore

import os  # type: ignore

import pandas as pd  # type: ignore

from traitlets import Float, Int, Unicode  # type: ignore

from typing import Dict


# -


# ## UI components

# For each column in ..., we define a cell widget that visualizes the contents depending on the column type. A cell widget can have an associated viewer if further visualization is required.

# ### Cell widgets


class CellWidget:
    """Base class for a cell widget which displays data depending on the column type. Some cell widgets might require additional viewers."""

    def __init__(self):
        self._viewer = None

    @property
    def viewer(self):
        """Returns the special viewer for this widget.
        
        Returns
        -------
             the special viewer for this widget, `None` if no special viewer is required.
        """
        return self._viewer


class LabelCellWidget(Label, CellWidget):
    """A cell widget for data type `str` that simply displays the given string.
    
    Requires no additional viewer.
    """

    def __init__(self, *args, **kwargs):
        CellWidget.__init__(self)
        Label.__init__(self, *args, **kwargs)


class ExplicitVBRCellWidget(CellWidget, Checkbox):
    """ A cell widget for data type `ExplicitVBR` that displays a checkbox connected to a viewer that visualizes spatial image of `ExplicitVBR`.
    """

    def __init__(self, obj: ExplicitVBR, *args, **kwargs):
        """Initializes the widget with the specified `obj`.
        
        Parameters
        ----------
        obj: ExplicitVBR
            
        """
        CellWidget.__init__(self,)
        Checkbox.__init__(self, *args, **kwargs)

        self.value = False
        self.description = "show region"

        # viewer that visualizes the spatial image when checkbox is checked.
        self._viewer = ViewerFactory.get_region_viewer()

        def selection_changed(change, image):
            if change["new"]:
                self._viewer.add(image)
            else:
                self._viewer.remove(image)

        self.observe(
            partial(selection_changed, image=obj.spatial_image()), names="value"
        )


# ### Custom cell widgets

# #### Link widget

# A custom link widget to display links.


@register
class LinkWidget(DOMWidget):
    _view_name = Unicode("LinkView").tag(sync=True)
    _view_module = Unicode("link_widget").tag(sync=True)
    _view_module_version = Unicode("0.1.0").tag(sync=True)

    # value to appear as link
    value = Unicode().tag(sync=True)
    # url of the link
    href = Unicode().tag(sync=True)


# + language="javascript"
# require.undef('link_widget');
#
# define('link_widget', ["@jupyter-widgets/base"], function(widgets) {
#
#     var LinkView = widgets.DOMWidgetView.extend({
#
#         // Render the view.
#         render: function() {
#             this.link = document.createElement('a');
#             this.link.setAttribute('target', '_blank');
#
#             this.link.setAttribute('href', this.model.get('href'));
#             this.link.innerHTML = this.model.get('value');
#
#             this.el.appendChild(this.link);
#         },
#
#         value_changed: function() {
#             this.link.setAttribute('href', this.model.get('href'));
#             this.link.innerHTML = this.model.get('value');
#         }
#
#     });
#
#     return {
#         LinkView: LinkView
#     };
# });
# -


class StudyIdWidget(CellWidget, LinkWidget):
    """A widget to display PubMed study IDs as links to publications."""

    __URL = "https://www.ncbi.nlm.nih.gov/pubmed/?term="
    __PubMed = "PubMed"

    def __init__(self, study_id, *args, **kwargs):
        """
        Parameters
        ----------
        study_id : str, StudyID
            PubMed study ID.
        """
        CellWidget.__init__(self)
        LinkWidget.__init__(
            self,
            value=StudyIdWidget.__PubMed + ":" + study_id,
            href=StudyIdWidget.__URL + study_id,
            *args,
            **kwargs,
        )


a = StudyIdWidget("23773060")
a


# #### Progress widget

# A custom progress widget to display progress/percentage.


@register
class ProgressWidget(DOMWidget):
    _view_name = Unicode("ProgressView").tag(sync=True)
    _view_module = Unicode("progress_widget").tag(sync=True)
    _view_module_version = Unicode("0.1.0").tag(sync=True)

    # actual value
    value = Float().tag(sync=True)
    # maximum value
    max = Int().tag(sync=True)


# + language="javascript"
# require.undef('progress_widget');
#
# define('progress_widget', ["@jupyter-widgets/base"], function(widgets) {
#
#     var ProgressView = widgets.DOMWidgetView.extend({
#
#         // Render the view.
#         render: function() {
#             this.progress = document.createElement('progress');
#             this.progress.setAttribute('value',  this.model.get('value'));
#             // TODO set number of decimal places to display
#             this.progress.setAttribute('title',  this.model.get('value'));
#             this.progress.setAttribute('max', this.model.get('max'));
#
#             this.el.appendChild(this.progress);
#         },
#
#         value_changed: function() {
#             this.progress.setAttribute('value',  this.model.get('value'));
#             this.progress.setAttribute('title',  this.model.get('value'));
#             this.progress.setAttribute('max', this.model.get('max'));
#         }
#
#     });
#
#     return {
#         ProgressView: ProgressView
#     };
# });
# -


class TfIDfWidget(CellWidget, ProgressWidget):
    """A widget to display TfIDf value ."""

    def __init__(self, tfidf, *args, **kwargs):
        """
        Parameters
        ----------
        tfidf : float, TfIDf
            .
        """
        CellWidget.__init__(self)
        ProgressWidget.__init__(self, value=tfidf, max=1, *args, **kwargs)


a = TfIDfWidget(0.23589651054299998)
a

# ### Cell viewer widgets


# Cell viewer widgets visualize data of a columntype in a separate area.


def encode_images(images):
    encoded_images = []
    image_txt = []
    for i, image in enumerate(images):
        nifti_image = nib.Nifti2Image(image.get_fdata(), affine=image.affine)
        encoded_image = base64.encodebytes(nifti_image.to_bytes())
        del nifti_image
        image_txt.append(f"image{i}")
        enc = encoded_image.decode("utf8").replace("\n", "")
        encoded_images.append(f'var {image_txt[-1]}="{enc}";')

    encoded_images = "\n".join(encoded_images)
    return encoded_images, image_txt


class PapayaViewerWidget(HTML):
    """A viewer that overlays multiple label maps.
    
    Number of label maps to overlay is limited to 8. ??
    """

    encoder = json.JSONEncoder()

    # html necessary to embed papaya viewer
    papaya_html = """
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

    def __init__(self, atlas="avg152T1_brain.nii.gz", *args, **kwargs):
        """Initializes the widget with the specified `atlas`.
        
        Parameters
        ----------
        atlas: str
            path for the image file to be used as atlas.
        """
        HTML.__init__(self, *args, **kwargs)

        # load atlas and add it to image list
        self.atlas_image = nib.load(atlas)
        self.images = [self.atlas_image]

        # papaya parameters
        self.params = {"kioskMode": False, "worldSpace": True, "fullScreen": False}

        # initially plot the atlas
        self.plot()

    def add(self, image):
        """Adds the specified `image` to the image list of this viewer.
        
        Parameters
        ----------
        image:
            image to be added to the list of this viewer.
        """
        self.images.append(image)
        self.plot()

    def remove(self, image):
        """Removes the specified `image` from the image list of this viewer.
        
        Parameters
        ----------
        image:
            image to be removed from the image list of this viewer.
        """
        self.images.remove(image)
        self.plot()

    def plot(self, center_image=None):
        """Plots all images in the image list of this viewer.
        
        Parameters
        ----------
        center_image:
            the image to center the view.
        
        Note
        ----
        As papaya has a limit of 8 images, it can display only 8 images overlaid. Selection of images depends on the implementation of papaya.
        """

        # set center_image as the last appended image if not specified
        if center_image == None and len(self.images) > 0:
            center_image = self.images[-1]

        # encode images
        encoded_images, image_names = encode_images(self.images)

        # set params variable for papaya
        params = dict()
        params.update(self.params)
        params["encodedImages"] = image_names

        for image_name in image_names[1:]:
            params[image_name] = {"min": 0, "max": 10, "lut": "Red Overlay"}

        if center_image is not None:
            coords = (
                np.transpose(center_image.get_fdata().nonzero()).mean(0).astype(int)
            )
            coords = nib.affines.apply_affine(center_image.affine, coords)
            params["coordinate"] = [int(c) for c in coords]

        escaped_papaya_html = html.escape(
            PapayaViewerWidget.papaya_html.format(
                params=PapayaViewerWidget.encoder.encode(params),
                encoded_images=encoded_images,
            )
        )
        iframe = (
            f'<iframe srcdoc="{escaped_papaya_html}" id="papaya" '
            f'width="700px" height="600px"></iframe>'
        )
        self.value = iframe


# ###  Factories


class CellWidgetFactory:
    @staticmethod
    def get_widget(obj):
        if isinstance(obj, neurolang.frontend.ExplicitVBR):
            return ExplicitVBRCellWidget(obj)
        elif isinstance(obj, neurolang.frontend.neurosynth_utils.StudyID):
            studyid = str(obj)
            return StudyIdWidget(studyid)
        elif isinstance(obj, neurolang.frontend.neurosynth_utils.TfIDf):
            return TfIDfWidget(float(obj))
        # TODO remove this case when TfIDf is added as column.
        elif isinstance(obj, float):
            return TfIDfWidget(float(obj))
        else:
            return LabelCellWidget(str(obj))


class ViewerFactory:
    papaya_viewer = PapayaViewerWidget(
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
        # TODO this is to avoid performance problems until paging is implemented
        nb_rows = min(len(wras), 20)
        table = sheet(
            rows=nb_rows,
            columns=wras.arity,
            column_headers=column_headers,
            layout=Layout(width="auto", height=f"{(50 * rows_visible) + 10}px"),
        )

        for i, tuple_ in enumerate(wras.unwrapped_iter()):
            row_temp = []
            for el in tuple_:
                cell_widget = CellWidgetFactory.get_widget(el)
                row_temp.append(cell_widget)
                if cell_widget.viewer is not None:
                    self.cell_viewers.add(cell_widget.viewer)
            row(i, row_temp)
            # TODO this is to avoid performance problems until paging is implemented
            if i == nb_rows - 1:
                break
        return table


class ResultWidget(VBox):
    def __init__(self):
        super(ResultWidget, self).__init__()

        self.viewers = set()

    def show_results(self, res: Dict[str, WrappedRelationalAlgebraSet]):
        tablesets = self._create_tablesets(res)

        self.children = tuple(self.viewers) + tuple(tablesets)

    def _create_tablesets(self, res):

        answer = "ans"

        tablesets = []
        for name, result_set in res.items():
            tableset_widget = TableSetWidget(name, result_set, self.viewers)

            if name == answer:
                tablesets.insert(0, tableset_widget)
            else:
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


# ### Init agent


def init_agent():
    """
    Set up the neurolang query runner (?) and add facts (?) to
    the database
    """
    nl = NeurolangDL()

    @nl.add_symbol
    def region_union(rs):
        return regions.region_union(rs)

    # TODO this can be removed after the bug is fixed
    # currently symbols are listed twice
    nl.reset_program()

    return nl


def run_query(nl, query):
    with nl.scope as s:
        nl.execute_nat_datalog_program(query)
        return nl.solve_all()


# ### Frontend feed functions


def add_destrieux(nl):
    destrieux = nl.new_symbol(name="destrieux")
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


def add_subramarginal(nl):
    nl.load_neurosynth_term_regions("supramarginal", name="neurosynth_supramarginal")


def add_def_mode_study(nl):
    nl.load_neurosynth_term_study_ids(
        term="default mode", name="neurosynth_default_mode_study_id"
    )


def add_pcc_study(nl):
    nl.load_neurosynth_term_study_ids(term="pcc", name="neurosynth_pcc_study_id")


def add_study_tf_idf(nl):
    nl.load_neurosynth_study_tfidf_feature_for_terms(
        terms=["default mode", "pcc"], name="neurosynth_study_tfidf",
    )


# ## Query the NeuroLang engine and display results

nl = init_agent()

add_destrieux(nl)

add_subramarginal(nl)

add_def_mode_study(nl)

add_pcc_study(nl)

add_study_tf_idf(nl)

# +
default_query = "ans(region_union(r)) :- destrieux(..., r)"
query = "".join(
    "ans(study_id, term, tfidf):-neurosynth_default_mode_study_id(study_id),"
    "neurosynth_pcc_study_id(study_id),"
    "neurosynth_study_tfidf(study_id, term, tfidf)"
)

qw = QueryWidget(nl, query)
qw
