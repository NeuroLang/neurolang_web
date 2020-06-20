from ipysheet import row, sheet  # type: ignore
from ipywidgets import Button, HBox, HTML, Layout, VBox  # type: ignore

from neurolang.datalog.wrapped_collections import (
    WrappedRelationalAlgebraSet,
)  # type: ignore

from neurolang_ipywidgets import NlCodeEditor, NlIconTab
from nlweb.viewers.factory import ColumnsManager

# This should be changed when neurolang gets
# a unified exceptions hierarchy
from tatsu.exceptions import FailedParse

from traitlets import Unicode  # type: ignore

from typing import Dict


class ResultTabWidget(VBox):
    """"""

    icon = Unicode()

    def __init__(self, name: str, wras: WrappedRelationalAlgebraSet, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            layout=Layout(width="auto", height=f"{(50 * rows_visible) + 30}px"),
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

        for viewer in viewers:
            viewer.reset()

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
            tableset_widget = ResultTabWidget(
                name, result_set, layout=Layout(height="340px")
            )

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
    """
    A widget to input queries

    Parameters
    ----------

    neurolang_engine: NeurolangDL
                      Engine to query
    default_query: str
                   Default query text, will be shown in textarea
    reraise: bool
             re-raise exceptions thrown during query execution
    """

    def __init__(
        self,
        neurolang_engine,
        default_query="ans(region_union(r)) :- destrieux(..., r)",
        reraise=False,
    ):
        super().__init__()

        # TODO check if neurolang_engine is None.

        self.neurolang_engine = neurolang_engine
        self.reraise = reraise

        self.query = NlCodeEditor(
            default_query,
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
        self.error_display = HTML(layout=Layout(visibility="hidden"))

        self.result_viewer = ResultWidget()

        self.children = [
            HBox([self.query, self.button]),
            self.error_display,
            self.result_viewer,
        ]

    def run_query(self, query: str):
        with self.neurolang_engine.scope:
            self.neurolang_engine.execute_datalog_program(query)
            return self.neurolang_engine.solve_all()

    def _on_query_button_clicked(self, b):
        """Runs the query in the query text area and diplays the results.

        Parameters
        ----------
        b: ipywidgets.Button
            button clicked.
        """

        self._reset_output()

        try:
            qresult = self.run_query(self.query.text)
        except FailedParse as fp:
            self._set_error_marker(fp)
            self._handle_generic_error(fp)
        except Exception as e:
            self.handle_generic_error(e)
        else:
            self.result_viewer.layout.visibility = "visible"
            self.result_viewer.show_results(qresult)

    def _reset_output(self):
        self.query.clear_marks()
        self.result_viewer.reset()
        self.result_viewer.layout.visibility = "hidden"
        self.error_display.layout.visibility = "hidden"

    def _set_error_marker(self, pe: FailedParse):
        try:
            line_info = pe.tokenizer.line_info(pe.pos)
        except AttributeError:
            # support tatsu 4.x
            line_info = pe.buf.line_info(pe.pos)

        self.query.marks = [{"line": line_info.line, "text": pe.message}]
        self.query.text_marks = [
            {
                "from": {"line": line_info.line, "ch": line_info.col},
                "to": {"line": line_info.line, "ch": line_info.col + 1},
            }
        ]

    def _handle_generic_error(self, e: Exception):
        self.error_display.layout.visibility = "visible"
        self.error_display.value = _format_exc(e)
        if self.reraise:
            raise e


def _format_exc(e: Exception):
    """
    Format an exception for display
    """
    return f"<pre style='background-color:#faaba5; border: 1px solid red; padding: 0.4em'>{e}</pre>"
