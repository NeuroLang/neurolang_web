# -*- coding: utf-8 -*-
# ## Neurolang Voilà UI

import warnings  # type: ignore

warnings.filterwarnings("ignore")

# +
from ipysheet import row, sheet  # type: ignore
from ipywidgets import (
    Button,
    HBox,
    HTML,
    Layout,
    Textarea,
    VBox,
)  # type: ignore

import neurolang
from neurolang import regions  # type: ignore
from neurolang.datalog.wrapped_collections import (
    WrappedRelationalAlgebraSet,
)  # type: ignore
from neurolang.frontend import NeurolangDL, ExplicitVBR  # type: ignore

from nlweb.viewers import PapayaViewerWidget
from nlweb.viewers.cell import NlIconTab

import nibabel as nib

from nilearn import datasets  # type: ignore

from traitlets import Unicode  # type: ignore

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
