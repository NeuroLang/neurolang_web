import html
import gzip

from ipysheet import column, hold_cells, sheet  # type: ignore
from ipywidgets import (
    BoundedIntText,
    Button,
    HBox,
    HTML,
    Label,
    Layout,
    Select,
    Tab,
    Text,
    VBox,
    Output,
)  # type: ignore

from math import ceil

from neurolang.utils.relational_algebra_set.pandas import NamedRelationalAlgebraFrozenSet  # type: ignore

from neurolang_ipywidgets import NlCodeEditor, NlDownloadLink, NlIconTab


# This should be changed when neurolang gets
# a unified exceptions hierarchy
from tatsu.exceptions import FailedParse

from traitlets import Int, Unicode  # type: ignore

from typing import Callable, Dict, Optional, Tuple

from nlweb.util import debounce
from nlweb.viewers.factory import ViewerFactory, ColumnsManager


class PaginationWidget(HBox):
    """A pagination widget that enables setting page number and the number of rows per page."""

    # number of rows in a page by default
    DEFAULT_LIMIT = 50
    # max number of rows to avoid performance problems
    MAX_LIMIT = 100

    # current page number
    page = Int()
    # number of rows per page.
    limit = Int()

    def __init__(self, nb_rows, limit=50, *args, **kwargs):
        """
        Parameters
        ----------
        nb_rows: int
            total number of rows in the result set.
        limit: int
            number of rows to display in a page.

        """
        super().__init__(*args, **kwargs)

        self.__nb_rows = nb_rows if nb_rows else 1
        self.page = 1
        self.limit = (
            limit
            if limit and limit > 0 and limit < PaginationWidget.MAX_LIMIT
            else PaginationWidget.DEFAULT_LIMIT
        )

        self.layout.width = "400px"

        if nb_rows <= limit:
            self.layout.visibility = "hidden"
        else:
            self.layout.visibility = "visible"

            nb_pages = self._get_nb_pages(self.limit)

            # widget to set page number
            self.__page_widget = BoundedIntText(
                value=self.page,
                min=1,
                max=nb_pages,
                step=1,
                continuous_update=True,
                description="page",
                description_tooltip="Current page",
                disabled=False,
                style={"description_width": "30px"},
                layout=Layout(width="90px", max_width="90px"),
            )

            # widget to display total number of pages.
            self.__label_slash = Label(
                value=f"/ {nb_pages}", layout=Layout(width="60px")
            )

            # widget to set limit
            self.__limit_widget = BoundedIntText(
                value=self.limit,
                min=1,
                max=PaginationWidget.MAX_LIMIT,
                step=1,
                continuous_update=True,
                description="rows",
                description_tooltip=f"Number of rows per page. Max. possible: {PaginationWidget.MAX_LIMIT}",
                disabled=False,
                style={"description_width": "30px"},
                layout=Layout(width="90px", max_width="90px"),
            )

            self.__page_widget.observe(self._page_widget_changed, names="value")
            self.__limit_widget.observe(self._limit_widget_changed, names="value")

            self.children = [
                self.__page_widget,
                self.__label_slash,
                self.__limit_widget,
            ]

    def _get_nb_pages(self, limit):
        return ceil(self.__nb_rows / limit)

    def _page_widget_changed(self, change):
        self.page = change["new"]

    def _limit_widget_changed(self, change):
        new_limit = change["new"]
        # update limit
        self.limit = new_limit
        self.page = 1

        nb_pages = self._get_nb_pages(new_limit)

        # update page widget
        self.__page_widget.max = nb_pages
        self.__page_widget.value = 1

        # update label slash widget
        self.__label_slash.value = f"/ {nb_pages}"


class GraphOutputWidget(VBox):

    icon = Unicode()

    def __init__(
        self,
        title: str,
        res: Dict[str, NamedRelationalAlgebraFrozenSet],
        callback: Callable,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.loaded = False
        self.res = res
        self.callback = callback
        self._args = args
        self._kwargs = kwargs

        self._output = Output()
        self._output.layout = Layout(width="100%", min_height="400px")

    def load(self):
        if not self.loaded:
            self.loaded = True

            self._load_output()

            self.children = [self._output]

    def _load_output(self):

        with self._output:
            try:
                self.callback(self.res, *self._args, **self._kwargs)
            except Exception as e:
                print(f"An error occurred while producing this output.\n{e}")


class ResultTabPageWidget(VBox):
    """Tab page widget that displays result table and controls for each column type in the result table."""

    icon = Unicode()

    DOWNLOAD_THRESHOLD = 500000

    def __init__(
        self,
        title: str,
        nras: NamedRelationalAlgebraFrozenSet,
        viewer_factory: ViewerFactory,
        *args,
        **kwargs,
    ):
        """

        Parameters
        ----------
        title: str
            title for the tab page.
        nras: NamedRelationalAlgebraFrozenSet
            query result for the specified `title`.
        viewer_factory: ViewerFactory
            viewer factory to get viewer for corresponding column type
        """
        super().__init__(*args, **kwargs)
        self.loaded = False
        self._df = nras.as_pandas_dataframe()

        try:
            self._df = self._df.sort_values(self._df.columns[0])
        except TypeError:
            # print(f"Table {title} cannot be sorted.")
            pass

        self._total_nb_rows = self._df.shape[0]

        # initialize columns manager that generates widgets for each column, column viewers, and controls
        self._columns_manager = ColumnsManager(self, nras.row_type, viewer_factory)

        self._cell_viewers = self._columns_manager.get_viewers()

        tab_controls = self._columns_manager.get_controls()
        self._hbox_title.children = self._create_title(title, tab_controls)

    def _create_title(self, title, tab_controls):
        """Creates title controls for this tab widget.

        - Adds title label
        - Adds download button for query result. Disabled if one of the following conditions hold:
            * query result contains ExplicitVBR or ExplicitVBROverlay type column
            * number of rows in the query result exceeds DOWNLOAD_THRESHOLD
        - Adds paginator if there exists no ExplicitVBR or ExplicitVBROverlay type column
        - Adds any controls related to column types in the result set

        Parameters
        ----------
        title: str
            result set title.
        tab_controls: list
            list of controls related to columns in the result set.

        """
        # initialize widgets
        # add title label
        title_label = HTML(
            f"<h3>{title}</h3>", layout=Layout(padding="0px 5px 5px 0px")
        )

        self._hbox_title = HBox(
            layout=Layout(justify_content="space-between", align_items="center")
        )

        # create download link
        dw = NlDownloadLink(
            layout=Layout(
                width="30px",
                max_width="30px",
                min_width="30px",
                margin="5px 5px 5px 0",
                padding="0 0 0 0",
                flex="0 1 0",
                align_self="center",
            )
        )

        hbox_table_info = HBox(
            [title_label, dw],
            layout=Layout(justify_content="flex-start", align_items="center"),
        )

        if not self._columns_manager.hasVBRColumn:

            if self._total_nb_rows <= ResultTabPageWidget.DOWNLOAD_THRESHOLD:
                dw.filename = f"{title}.csv.gz"
                dw.mimetype = "application/gz"
                dw.tooltip = f"Download {dw.filename} file."

                def clicked(event):
                    dw.content = gzip.compress(self._df.to_csv(index=False).encode())

                dw.on_click(clicked)
            else:
                dw.disabled = True
                dw.tooltip = "Not available for download due to size!"

            # add paginator if there exist no ExplicitVBR or ExplicitVBROverlay column
            paginator = PaginationWidget(
                self._df.shape[0], layout=Layout(padding="0px 0px 0px 50px")
            )
            self._limit = paginator.limit
            paginator.observe(self._page_number_changed, names="page")
            paginator.observe(self._limit_changed, names="limit")

            hbox_table_info.children = hbox_table_info.children + (paginator,)
        else:
            dw.tooltip = "Not available for download due to column type!"
            dw.disabled = True

            self._limit = self._total_nb_rows

        return [hbox_table_info, HBox(tab_controls)]

    def load(self):
        if not self.loaded:
            self.loaded = True

            self._load_table(1, self._limit)

            self.children = [self._hbox_title, self._table]

    def _load_table(self, page, limit):
        """
        Parameters
        ----------
        page: int
            page number to view.
        limit: int
            number of rows to display per page.
        """

        start = (page - 1) * limit

        if start < 0 or start >= self._total_nb_rows:
            raise ValueError(
                f"Specified page number {page} and limit {limit} are not valid for result set of {self._total_nb_rows} rows."
            )

        end = min(start + limit, self._total_nb_rows)

        number_of_rows = end - start

        self._table = sheet(
            rows=min(self._total_nb_rows, number_of_rows),
            columns=len(self._df.columns),
            column_headers=list(self._df.columns),
            layout=Layout(width="auto", height="330px"),
        )

        with hold_cells():
            for col_index, column_id in enumerate(self._df.columns):
                column_data = self._df[column_id]
                column_feeder = self._columns_manager.get_column_feeder(col_index)
                rows = []

                for row_index in range(start, end):
                    rows.append(column_feeder.get_widget(column_data.iloc[row_index]))
                    column(col_index, rows, row_start=0)

    @debounce(0.5)
    def _page_number_changed(self, change):
        page_number = change["new"]

        self._load_table(page_number, self._limit)
        self.children = [self._hbox_title, self._table]

    @debounce(0.5)
    def _limit_changed(self, change):
        self._limit = change["new"]

        self._load_table(1, self._limit)
        self.children = [self._hbox_title, self._table]

    def get_viewers(self):
        """Returns list of viewers for this tab page.

        list
            list of cell viewers for this tab page.
        """
        return self._cell_viewers


class QResultWidget(VBox):
    """A widget to display query results and corresponding viewers."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # tab widget that displays each resultset in an individual tab
        self._tab = NlIconTab(layout=Layout(height="460px"))
        # viewers necessary for each resultset, can be shared among resultsets
        self._viewers = None

    def _create_result_tabs(
        self,
        res: Dict[str, NamedRelationalAlgebraFrozenSet],
        callbacks: Dict[str, Callable] = None,
    ):
        """Creates necessary tab pages and viewers for the specified query result `res`.

        Parameters
        ----------
        res: Dict[str, NamedRelationalAlgebraFrozenSet]
           dictionary of query results with keys as result name and values as result for corresponding key.
        callbacks : Dict[str, Callable], optional
            dict of callback functions to display outputs in result tabs, by default None

        Returns
        -------
        result_tabs: list
            list of tab pages to be added to tab as children.
        titles: list
            list of titles for tab pages.
        icons: list
            list of icons for tab pages.
        viewers: set
            set of viewers for all tab pages.
        """
        result_tabs = []
        titles = []
        icons = []

        # to be passed to each tab page to use viewers from the same factory
        viewer_factory = ViewerFactory()

        # set of all viewers for each result_tab
        viewers = set()

        def icon_changed(change):
            icons = []

            for result_tab in result_tabs:
                icons.append(result_tab.icon)
            self._tab.title_icons = icons

        for name in sorted(res.keys()):
            result_set = res[name]
            result_tab = ResultTabPageWidget(
                name, result_set, viewer_factory, layout=Layout(height="100%")
            )

            result_tabs.append(result_tab)
            titles.append(name)
            icons.append(result_tab.icon)

            result_tab.observe(icon_changed, names="icon")

            viewers = viewers | result_tab.get_viewers()

        if callbacks is not None:
            for name, callback in callbacks.items():
                output_tab = GraphOutputWidget(name, res, callback)
                result_tabs.append(output_tab)
                titles.append(name)
                icons.append(output_tab.icon)

        return result_tabs, titles, icons, viewers

    def show_results(
        self,
        res: Dict[str, NamedRelationalAlgebraFrozenSet],
        default_symbol: str,
        callbacks: Dict[str, Callable] = None,
    ):
        """Creates and displays necessary tab pages and viewers for the specified query result `res`.

        Parameters
        ----------
        res: Dict[str, NamedRelationalAlgebraFrozenSet]
           dictionary of query results with keys as result name and values as result for corresponding key.
        default_symbol: str
            the key for the result tab to display by default
        callbacks : Dict[str, Callable], optional
            dictonary of titles -> callback functions which will be called to create an output in one of the
            result tabs, by default None
        """

        result_tabs, titles, icons, self._viewers = self._create_result_tabs(
            res, callbacks
        )

        self._tab.children = result_tabs
        selected_index = 0
        for i, title in enumerate(titles):
            self._tab.set_title(i, title)
            if title == default_symbol:
                selected_index = i

        self._tab.title_icons = icons

        # observe to load each table upon tab selection
        self._tab.observe(self._tab_index_changed, names="selected_index")

        # select default tab so that data is loaded and it is viewed initially
        self._tab.selected_index = selected_index

        self.children = (self._tab,) + tuple(self._viewers)

    def _tab_index_changed(self, change):
        """Loads the result table for the selected tab."""
        tab_page = self._tab.children[self._tab.selected_index]

        if not tab_page.loaded:
            tab_page.load()

    def reset(self):
        """Resets this query result widget removing all tabs in tab widget and resetting and removing all viewers."""
        if self._viewers is not None:
            for viewer in self._viewers:
                viewer.reset()
        self._viewers = None

        self._tab.reset()


class SymbolsWidget(HBox):
    """
    A list of symbols, plus a filtering search box
    """

    def __init__(self, nl, **kwargs):
        self.nl = nl
        self.list = Select(options=self.nl.symbols)
        self.search_box = Text(placeholder="search")
        self.help = HTML()
        super().__init__(**kwargs)

        self.children = [VBox([self.search_box, self.list]), self.help]

        self.help.layout = Layout(flex="1 1 65%")

        self.list.observe(self.on_select_change, names="value")
        self.on_select_change(None)

        self.search_box.observe(self.on_search_change, names="value")

    def on_select_change(self, change):
        help = self.nl.symbols[self.list.value].help()
        self.help.value = _format_help_message(self.list.value, help)

    def on_search_change(self, change):
        if self.search_box.value == "":
            self.list.options = self.nl.symbols
        else:
            filtered_options = [
                item for item in self.nl.symbols if self.search_box.value in item
            ]
            self.list.options = filtered_options


_help_message_style = """


<style >
  .help-section {
    margin-left: 5px;}

  .help-header {
    background: lightGray;
    border-bottom: 1px solid black;}

  .help-body {
    padding-left: 5px;
    padding-top: 5px;}

  .unavailable {
    background: lightyellow;}
</style >
"""


def _format_help_message(symbol: str, help: Optional[str]) -> str:
    body = (
        f"<pre>{html.escape(help)}</pre>"
        if help is not None
        else "<p class='unavailable'>help unavailable</p>"
    )

    markup = f"""
    {_help_message_style}
    <div class = "help-section" >
      <p class = "help-header" >
        <i class = "fa fa-fw fa-question-circle" aria-hidden = "true" > </i > help for <b> {symbol} </b>
      </p >
      <div class = "help-body" >
        {body}
      </div >
    </div >
    """
    return markup


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
    callbacks : Dict[str, Callable], optional
        dictionnary of title -> callback function, by default None
    """

    def __init__(
        self,
        neurolang_engine,
        default_query="union(region_union(r)) :- destrieux(..., r)",
        reraise=False,
        callbacks: Dict[str, Callable] = None,
    ):
        if neurolang_engine is None:
            raise TypeError("neurolang_engine should not be NoneType!")

        super().__init__()

        self.layout.max_width = "1000px"

        self.neurolang_engine = neurolang_engine
        self.reraise = reraise
        self.callbacks = callbacks

        self.query = NlCodeEditor(
            default_query,
            disabled=False,
            layout=Layout(
                display="flex",
                flex_flow="row",
                align_items="stretch",
                width="75%",
                min_height="100px",
                border="solid 1px silver",
            ),
        )
        self.button = Button(description="Run query")
        self.button.on_click(self._on_query_button_clicked)
        self.error_display = HTML(layout=Layout(visibility="hidden"))
        self.info_display = HTML(layout=Layout(visibility="hidden"))
        self.query_section = Tab(
            children=[
                VBox(
                    [
                        HBox([self.query, self.button]),
                        self.error_display,
                        self.info_display,
                    ]
                ),
                SymbolsWidget(self.neurolang_engine),
            ]
        )
        for i, tab_title in enumerate(["query", "symbols"]):
            self.query_section.set_title(i, tab_title)

        self.result_viewer = QResultWidget(layout=Layout(visibility="hidden"))

        self.children = [self.query_section, self.result_viewer]

    def run_query(
        self, query: str
    ) -> Tuple[Dict[str, NamedRelationalAlgebraFrozenSet], str]:
        with self.neurolang_engine.scope:
            query_res = self.neurolang_engine.execute_datalog_program(query)
            last_symbol = None
            if query_res is None:
                # There is no query rule in the program, run solve_all
                res = self.neurolang_engine.solve_all()
                # Try to find the name of the last symbol in the program to display it by default
                last_rule = str(self.neurolang_engine.current_program[-1])
                for s in res.keys():
                    if last_rule.startswith(s) and (
                        last_symbol is None or len(s) > len(last_symbol)
                    ):
                        last_symbol = s
            else:
                # There was a query in the program, return a dict with just the result_set
                res = {"ans": query_res}
                last_symbol = "ans"

            return res, last_symbol

    def _on_query_button_clicked(self, b):
        """Runs the query in the query text area and diplays the results.

        Parameters
        ----------
        b: ipywidgets.Button
            button clicked.
        """

        self._reset_output()

        try:
            self._display_info("Your query is running, this may take a while ...")
            qresult, default_symbol = self.run_query(self.query.text)
        except FailedParse as fp:
            self._set_error_marker(fp)
            self._handle_generic_error(fp)
        except Exception as e:
            self._handle_generic_error(e)
        else:
            if qresult != {}:
                self.result_viewer.show_results(qresult, default_symbol, self.callbacks)
                self.result_viewer.layout.visibility = "visible"
            else:
                self._handle_generic_error(
                    ValueError("Query did not return any results.")
                )
        finally:
            self._display_info()

    def _reset_output(self):
        self.error_display.layout.visibility = "hidden"
        self.info_display.layout.visibility = "hidden"
        self.result_viewer.layout.visibility = "hidden"
        self.query.clear_marks()
        self.result_viewer.reset()

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

    def _display_info(self, info: str = None):
        if info is None:
            self.info_display.layout.visibility = "hidden"
        else:
            self.info_display.layout.visibility = "visible"
            self.info_display.value = _format_info(info)


def _format_exc(e: Exception):
    """
    Format an exception for display
    """
    return f"<pre style='background-color:#faaba5; border: 1px solid red; padding: 0.4em'>{e}</pre>"


def _format_info(info: str):
    """
    Format an info string for display
    """
    return f"<pre style='background-color:#a5affa; border: 1px solid blue; padding: 0.4em'>{info}</pre>"
