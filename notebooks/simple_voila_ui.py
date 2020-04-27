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
    ButtonStyle,
    Checkbox,
    DOMWidget,
    HBox,
    HTML,
    Label,
    Layout,
    Tab,
    Textarea,
    VBox,
    Widget,
    register,
)  # type: ignore

import json  # type: ignore

import neurolang
from neurolang import regions  # type: ignore
from neurolang.datalog.wrapped_collections import (
    WrappedRelationalAlgebraSet,
)  # type: ignore
from neurolang.frontend import NeurolangDL, ExplicitVBR  # type: ignore
from neurolang.frontend.neurosynth_utils import StudyID, TfIDf

from neurolang_ipywidgets import NlLink, NlProgress, NlCheckbox, NlIconTab

from nilearn import datasets  # type: ignore
import nibabel as nib  # type: ignore

import numpy as np  # type: ignore

import os  # type: ignore

import pandas as pd  # type: ignore

from traitlets import Float, Int, Unicode  # type: ignore

from typing import Dict


# -


# ## Query Agent


# ### Functions


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


# +
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
        terms=["default mode", "pcc"], name="neurosynth_study_tfidf"
    )


# -

# ### Prepare engine

nl = init_agent()
add_destrieux(nl)
add_subramarginal(nl)
add_def_mode_study(nl)
add_pcc_study(nl)
add_study_tf_idf(nl)

# ## UI components

# For each column in ..., we define a cell widget that visualizes the contents depending on the column type. A cell widget can have an associated viewer if further visualization is required.

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


class ColumnViewer:
    pass


class PapayaViewerWidget(HTML, ColumnViewer):
    """A viewer that overlays multiple label maps.
    
    Number of label maps to overlay is limited to 8. ??
    """

    encoder = json.JSONEncoder()

    # papaya parameters
    params = {"kioskMode": False, "worldSpace": True, "fullScreen": False}

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
        super().__init__(*args, **kwargs)

        # load atlas and add it to image list
        self.atlas_image = nib.load(atlas)
        self.images = [self.atlas_image]
        self._center = None
        self._center_coords = None

        # initially plot the atlas
        self.plot()

    def add(self, images):
        """Adds the specified `image` to the image list of this viewer.
        
        Parameters
        ----------
        images: list
            images to be added to the list of this viewer.
        """
        for image in images:
            self.images.append(image)
        self.plot()

    def remove(self, images):
        """Removes the specified `images` from the image list of this viewer.
        
        Parameters
        ----------
        images: list
            image to be removed from the image list of this viewer.
        """

        for image in images:
            self.images.remove(image)
        self.plot()

    def set_center(self, widget, image):
        """"""
        if self._center is not None:
            self._center.remove_center()
            self._center_coords = None

        # think of this
        if image is not None:
            self._center = widget
            self._center_coords = PapayaViewerWidget.calculate_coords(image)
        self.plot()

    def plot(self):
        """Plots all images in the image list of this viewer.
        
        Note
        ----
        As papaya has a limit of 8 images, it can display only 8 images overlaid. Selection of images depends on the implementation of papaya.
        """

        # set center_image as the last appended image if not specified
        if self._center is None and len(self.images) > 0:
            self._center_coords = PapayaViewerWidget.calculate_coords(self.images[-1])

        # encode images
        encoded_images, image_names = encode_images(self.images)

        # set params variable for papaya
        params = dict()
        params.update(PapayaViewerWidget.params)
        params["encodedImages"] = image_names
        if self._center_coords is not None:
            params["coordinate"] = self._center_coords

        for image_name in image_names[1:]:
            params[image_name] = {"min": 0, "max": 10, "lut": "Red Overlay"}

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

    @staticmethod
    def calculate_coords(image):
        """Calculates coordinates for the specified `image`."""
        coords = np.transpose(image.get_fdata().nonzero()).mean(0).astype(int)
        coords = nib.affines.apply_affine(image.affine, coords)
        return [int(c) for c in coords]

    def reset(self):
        self.images = [self.atlas_image]
        self._center = None
        self.plot()


# ### Cell widgets that extend neurolang_ipywidgets


class CellWidget:
    pass


class StudyIdWidget(NlLink, CellWidget):
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
        super().__init__(
            value=StudyIdWidget.__PubMed + ":" + study_id,
            href=StudyIdWidget.__URL + study_id,
            *args,
            **kwargs,
        )


a = StudyIdWidget("23773060")
a


class TfIDfWidget(NlProgress, CellWidget):
    """A widget to display TfIDf value ."""

    def __init__(self, tfidf, *args, **kwargs):
        """
        Parameters
        ----------
        tfidf : float, TfIDf
            .
        """
        super().__init__(value=tfidf, max=1, *args, **kwargs)


a = TfIDfWidget(0.23589651054299998)
a

# ### Cell widgets that extend standard ipywidgets


class LabelCellWidget(Label, CellWidget):
    """A cell widget for data type `str` that simply displays the given string.
    
    Requires no additional viewer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExplicitVBRCellWidget(HBox, CellWidget):
    """ A cell widget for data type `ExplicitVBR` that displays a checkbox connected to a viewer that visualizes spatial image of `ExplicitVBR`.
    """

    def __init__(
        self,
        obj: neurolang.regions.ExplicitVBR,
        viewer: PapayaViewerWidget,
        *args,
        **kwargs,
    ):
        """Initializes the widget with the specified `obj`.
        
        Parameters
        ----------
        obj: neurolang.regions.ExplicitVBR
        
        viewer : PapayaViewerWidget
            
        """
        super().__init__(*args, **kwargs)

        # viewer that visualizes the spatial image when checkbox is checked.
        self._viewer = viewer

        self.__image = obj.spatial_image()

        self._region_checkbox = NlCheckbox(
            value=False,
            description="show region",
            layout=Layout(
                width="120px", margin="5px 15px 5px 0", padding="5px 15px 5px 15px"
            ),
        )
        self._region_checkbox.observe(
            partial(self._selection_changed, image=self.__image), names="value"
        )

        self._center_btn = Button(
            tooltip="Center on region", icon="map-marker", layout=Layout(width="30px")
        )
        self._center_btn.on_click(self._center_btn_clicked)
        self._centered = False

        self.layout.align_items = "center"

        self.children = [
            self._region_checkbox,
            self._center_btn,
        ]

    @property
    def image(self):
        return self.__image

    @property
    def is_region_selected(self):
        return self._region_checkbox.value

    def disable_region(self, is_disabled):
        self._region_checkbox.disabled = is_disabled
        self._center_btn.disabled = is_disabled

    def unselect_region(self):
        self._region_checkbox.value = False

    def _selection_changed(self, change, image):
        if change["new"]:
            self._viewer.add([image])
        else:
            self._viewer.remove([image])

    def center_region(self, is_centered):
        self._centered = is_centered
        if is_centered:
            self._center_btn.icon = "map-pin"
        else:
            self._center_btn.icon = "map-marker"

    def _center_btn_clicked(self, b):
        if not self._centered:
            self.center_region(True)
            if not self._region_checkbox.value:
                self._region_checkbox.value = True
            self._viewer.set_center(self, self.image)

    def remove_center(self):
        self.center_region(False)


# ### Columns


class ColumnFeeder:
    """Base class for a column feeder which works as a factory to create cell widgets and their corresponding controls and viewers of a specific type of column."""

    def __init__(self):
        self._viewer = None
        self._controls = []

    @property
    def viewer(self):
        """Returns the special viewer widget for this column.
        
        Returns
        -------
             the special viewer widget for this column, `None` if no special viewer is required.
        """
        return self._viewer

    @property
    def controls(self):
        """Returns list of widgets that are used to control the widgets of this column.
        
        Returns
        -------
        list 
            
        """
        return self._controls

    def get_widget(self, obj):
        """Returns a Label widget for the specified `obj`.
        
        Returns
        -------
        ipywidgets.widgets.Label
            
        """
        return LabelCellWidget(str(obj))


class ExplicitVBRColumn(ColumnFeeder):
    __ICON_ON = "eye"
    __ICON_OFF = "eye-slash"

    def __init__(self, result_tab):
        super().__init__()
        self.result_tab = result_tab

        self._viewer = ViewerFactory.get_region_viewer()

        self._turn_on_off_btn = Button(
            tooltip="Turn on/off selected regions",
            icon=ExplicitVBRColumn.__ICON_ON,
            layout=Layout(width="30px", padding_top="20px"),
        )
        self.result_tab.icon = self._turn_on_off_btn.icon

        self._turn_on_off_btn.on_click(self._on_turn_on_off_btn_clicked)
        self._controls.append(self._turn_on_off_btn)

        self._unselect_btn = Button(
            tooltip="Unselect all selected regions",
            description="Unselect All",
            layout=Layout(width="100px", padding_top="20px"),
        )
        self._unselect_btn.on_click(self._on_unselect_clicked)
        self._controls.append(self._unselect_btn)

        self._column_on = True

        self.__evbr_widget_list = []

    def get_widget(self, obj):
        """"""
        if isinstance(obj, neurolang.regions.ExplicitVBR):
            e_widget = ExplicitVBRCellWidget(obj, self._viewer)
            self.__evbr_widget_list.append(e_widget)
            return e_widget
        else:
            return LabelCellWidget(str(obj))

    def _selected_images(self):
        images = []
        for e_widget in self.__evbr_widget_list:
            if e_widget.is_region_selected:
                images.append(e_widget.image)
        return images

    def _on_unselect_clicked(self, b):
        images = []
        for e_widget in self.__evbr_widget_list:
            if e_widget.is_region_selected:
                e_widget.unselect_region()

    #                images.append(e_widget.image)
    #        self._viewer.remove(images)

    def _on_turn_on_off_btn_clicked(self, b):
        turn_off = (
            True
            if self._turn_on_off_btn.icon == ExplicitVBRColumn.__ICON_OFF
            else False
        )

        images = []
        for e_widget in self.__evbr_widget_list:
            e_widget.disable_region(self._column_on)
            if e_widget.is_region_selected:
                images.append(e_widget.image)

        if self._column_on:
            self._column_on = False
            self._turn_on_off_btn.icon = ExplicitVBRColumn.__ICON_OFF
            self._unselect_btn.disabled = True
            self._viewer.remove(images)
        else:
            self._viewer.add(images)
            self._column_on = True
            self._turn_on_off_btn.icon = ExplicitVBRColumn.__ICON_ON
            self._unselect_btn.disabled = False

        self.result_tab.icon = self._turn_on_off_btn.icon


class StudIdColumn(ColumnFeeder):
    def __init__(self):
        super().__init__()

    def get_widget(self, obj):
        return StudyIdWidget(str(obj))


class TfIDfColumn(ColumnFeeder):
    def __init__(self):
        super().__init__()

    def get_widget(self, obj):
        return TfIDfWidget(float(obj))


# ###  Factories


class ViewerFactory:
    papaya_viewer = PapayaViewerWidget(
        layout=Layout(width="700px", height="600px", border="1px solid black")
    )

    @staticmethod
    def get_region_viewer():
        return ViewerFactory.papaya_viewer


class ColumnFeederFactory:
    """A factory class that creates `ColumnFeeder`s for specified column types."""

    @staticmethod
    def get_column(result_tab, column_type):
        """Creates and returns a `ColumnFeeder` for the specified `column_type`.
        
        Parameters
        ----------
        result_tab: ResultTabWidget
            the result tab that views the required column.
        column_type: str
            type of the column for the required `ColumnFeeder`.
            
        Returns
        -------
        ColumnFeeder
            column feeder for the specified `column_type`.
                
        """
        if column_type == neurolang.regions.ExplicitVBR:
            return ExplicitVBRColumn(result_tab)
        elif column_type == neurolang.frontend.neurosynth_utils.StudyID:
            return StudIdColumn()
        elif (
            column_type == neurolang.frontend.neurosynth_utils.TfIDf
            or column_type == float
        ):
            return TfIDfColumn()
        else:
            return ColumnFeeder()


class ColumnsManager:
    """A class that creates column feeders for a specified `tuple` of column types and manages creation of widgets for each column and, their corresponding viewers and controls. """

    def __init__(self, result_tab, column_types: tuple):
        self.columns = []

        for column_type in column_types.__args__:
            self.columns.append(ColumnFeederFactory.get_column(result_tab, column_type))

    def get_cell_widget(self, index, obj):
        """Creates and returns the cell widget for the column specified by `index` and the object `obj` for that column.
        
        Parameters
        ----------
        index : int
            index of the column.
        obj : 
            object of column type at the specified `index` which will be used by the widget.
            
        Returns
        -------
        CellWidget
            a cell widget corresponding to the column type at `index` with the specified object `obj` of the same column type.
        """
        return self.columns[index].get_widget(obj)

    def get_viewers(self):
        """Iterates each column feeder to get their corresponding viewer widgets and returns the set of viewers.
        
        Returns
        -------
        set
            the set of viewer widgets for all columns.
        """
        viewers = set()
        for column in self.columns:
            if column.viewer is not None:
                viewers.add(column.viewer)
        return viewers

    def get_controls(self):
        """Iterates each column feeder to get their corresponding control widgets and returns the list of controls.
        
        Returns
        -------
        list
            the list of control widgets for all columns.
        """
        controls = []
        for column in self.columns:
            if column.controls is not None:
                controls.extend(column.controls)
        return controls


# ### Query and result widgets


class ResultTabWidget(VBox):
    """"""

    icon = Unicode()

    def __init__(self, name: str, wras: WrappedRelationalAlgebraSet):
        super().__init__()

        self.wras = wras

        # create widgets
        name_label = HTML(f"<h2>{name}</h2>")

        columns_manager = ColumnsManager(self, wras.row_type)

        self.sheet = self._init_sheet(self.wras, columns_manager)

        self.cell_viewers = columns_manager.get_viewers()

        self.controls = columns_manager.get_controls()

        if self.controls is not None:
            hbox_menu = HBox(self.controls)

            hbox = HBox([name_label, hbox_menu])
            hbox.layout.justify_content = "space-between"
            hbox.layout.align_items = "center"

            list_widgets = [hbox] + [self.sheet]
            self.children = tuple(list_widgets)
        else:
            self.children = [name_label, self.sheet]

    def _init_sheet(self, wras, columns_manager):
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
            for j, el in enumerate(tuple_):
                cell_widget = columns_manager.get_cell_widget(j, el)
                row_temp.append(cell_widget)
            row(i, row_temp)
            # TODO this is to avoid performance problems until paging is implemented
            if i == nb_rows - 1:
                break
        return table

    def get_viewers(self):
        return self.cell_viewers


class ResultWidget(VBox):
    """"""

    def __init__(self):
        super().__init__()
        self.tab = NlIconTab(layout=Layout(height="400px"))

    def show_results(self, res: Dict[str, WrappedRelationalAlgebraSet]):
        self.reset()
        names, tablesets, viewers, icons = self._create_tablesets(res)

        self.children = (self.tab,) + tuple(viewers)

        for i, name in enumerate(names):
            self.tab.set_title(i, name)

        self.tab.children = tablesets

        self.tab.title_icons = icons

    def _create_tablesets(self, res):
        answer = "ans"

        tablesets = []
        names = []
        icons = []

        viewers = set()

        for name, result_set in res.items():
            tableset_widget = ResultTabWidget(name, result_set)

            if name == answer:
                tablesets.insert(0, tableset_widget)
                names.insert(0, name)
                icons.insert(0, tableset_widget.icon)
            else:
                tablesets.append(tableset_widget)
                names.append(name)
                icons.append(tableset_widget.icon)

            viewers = viewers | tableset_widget.get_viewers()

            def icon_changed(change):
                icons = []

                for tableset in tablesets:
                    icons.append(tableset.icon)
                self.tab.title_icons = icons

            tableset_widget.observe(icon_changed, names="icon")

        return names, tablesets, viewers, icons

    def reset(self):
        self.tab = NlIconTab(layout=Layout(height="400px"))


class QueryWidget(VBox):
    """"""

    def __init__(
        self,
        neurolang_engine,
        default_query="ans(region_union(r)) :- destrieux(..., r)",
    ):
        super().__init__()

        # TODO check if neurolang_engine is None.

        self.neurolang_engine = neurolang_engine

        self.query = Textarea(
            value=default_query,
            placeholder="Type something",
            disabled=False,
            layout=Layout(
                display="flex",
                flex_flow="row",
                align_items="stretch",
                width="75%",
                height="100px",
            ),
        )
        self.button = Button(description="Run query")
        self.button.on_click(self._on_query_button_clicked)

        self.result_viewer = ResultWidget()

        self.children = [HBox([self.query, self.button]), self.result_viewer]

    def _on_query_button_clicked(self, b):
        """Runs the query in the query text area and diplays the results.
        
        Parameters
        ----------
        b: ipywidgets.Button
            button clicked.
        """

        self.result_viewer.reset()

        qresult = run_query(self.neurolang_engine, self.query.value)
        self.result_viewer.show_results(qresult)


# ## Query UI

# +
query = "".join(
    "ans(study_id, term, tfidf):-neurosynth_default_mode_study_id(study_id),"
    "neurosynth_pcc_study_id(study_id),"
    "neurosynth_study_tfidf(study_id, term, tfidf)"
)

qw = QueryWidget(nl, query)
qw
