# -*- coding: utf-8 -*-
# ## Neurolang Voilà UI

import warnings  # type: ignore

warnings.filterwarnings("ignore")

# +
from functools import partial

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

import neurolang
from neurolang import regions  # type: ignore
from neurolang.datalog.wrapped_collections import (
    WrappedRelationalAlgebraSet,
)  # type: ignore
from neurolang.frontend import NeurolangDL, ExplicitVBR  # type: ignore
from neurolang.frontend.neurosynth_utils import StudyID, TfIDf

from nlweb.viewers import CellWidget, PapayaViewerWidget
from nlweb.viewers.cell import NlCheckbox, NlIconTab, NlLink, NlProgress

import nibabel as nib

from nilearn import datasets  # type: ignore

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
# -
