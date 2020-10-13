import pytest
import asyncio

from neurolang.frontend import NeurolangDL

from nlweb.viewers.query import (
    PaginationWidget,
    QResultWidget,
    QueryWidget,
    ResultTabPageWidget,
)
from nlweb.viewers.column import ColumnFeeder


class TestQueryWidget:
    """Tests QueryWidget."""

    @pytest.fixture
    def widget(self, engine):
        return QueryWidget(neurolang_engine=engine)

    def test_create_engine_none(self):
        """Tests constructor with `None` value for `neurolang_engine`."""

        with pytest.raises(TypeError) as error:
            QueryWidget(neurolang_engine=None)
        assert error.value.args[0] == "neurolang_engine should not be NoneType!"

    def test_create_query_empty(self, engine):
        """Tests constructor with empty string specified for `default_query`."""

        widget = QueryWidget(neurolang_engine=engine, default_query="")
        assert widget.query.text == ""

    def test_create_query_none(self, engine):
        """Tests constructor with `None` specified for `default_query`."""

        widget = QueryWidget(neurolang_engine=engine, default_query=None)
        assert widget.query.text == ""

    def test_create_default_query(self, engine):
        """Tests constructor with no value specified for `default_query`."""
        widget = QueryWidget(neurolang_engine=engine)

        assert widget.neurolang_engine is not None
        assert widget.query is not None
        assert widget.query.text == "ans(region_union(r)) :- destrieux(..., r)"
        assert widget.result_viewer is not None
        assert widget.reraise == False
        assert widget.query_section is not None
        assert widget.error_display is not None

    def test_run_query(self, widget):
        """Tests run_query."""

        res = widget.run_query(widget.query.text)
        assert res == {}

    def test_query_button_clicked_empty_res(self, widget, monkeypatch):
        """Tests _query_button_clicked when empty result returns."""

        def mock_solve_all(*args, **kwargs):
            return {}

        monkeypatch.setattr(NeurolangDL, "solve_all", mock_solve_all)

        widget._on_query_button_clicked(None)
        assert widget.error_display.layout.visibility == "visible"
        assert (
            widget.error_display.value
            == f"<pre style='background-color:#faaba5; border: 1px solid red; padding: 0.4em'>{ValueError('Query did not return any results.')}</pre>"
        )
        assert widget.result_viewer.layout.visibility == "hidden"

    def test_query_button_clicked(self, widget, monkeypatch, mock_solve_all):
        """Tests _query_button_clicked when non-empty result returns."""

        monkeypatch.setattr(NeurolangDL, "solve_all", mock_solve_all)

        widget._on_query_button_clicked(None)
        assert widget.result_viewer.layout.visibility == "visible"
        assert widget.error_display.layout.visibility == "hidden"

    def test_query_button_clicked_query_parse_error(self, widget):
        """Tests _query_button_clicked when query FailedParse error occurs."""

        widget.query.text = "ans(study_id, term, tfidf):-neurosynth_default_mode_study_id(study_id),neurosynth_pcc_study_id(study_id),neurosynth_study_tfidf(study_id, term, tfidf"
        widget._on_query_button_clicked(None)
        assert widget.result_viewer.layout.visibility == "hidden"
        assert widget.error_display.layout.visibility == "visible"
        assert len(widget.query.marks) == 1
        assert len(widget.query.text_marks) == 1

    def test_reset_output(self, widget):
        """Tests _reset_output."""

        widget._on_query_button_clicked(None)
        widget._reset_output()
        assert widget.result_viewer._viewers == None
        assert widget.result_viewer.layout.visibility == "hidden"
        assert widget.error_display.layout.visibility == "hidden"
        assert len(widget.query.marks) == 0
        assert len(widget.query.text_marks) == 0

    def test_handle_generic_error(self, widget):
        """Tests _handle_generic_error."""
        widget._handle_generic_error(ValueError("Error"))
        assert (
            widget.error_display.value
            == f"<pre style='background-color:#faaba5; border: 1px solid red; padding: 0.4em'>{ValueError('Error')}</pre>"
        )


class TestQResultWidget:
    """Tests QResultWidget."""

    @pytest.fixture
    def widget(self):
        return QResultWidget()

    def test_create(self, widget):
        """Tests QResultWidget constructor."""

        assert widget._tab is not None
        assert widget._viewers is None

    def test_create_result_tabs(self, widget, res):
        """Tests QResultWidget _create_result_tabs."""

        result_tabs, titles, icons, viewers = widget._create_result_tabs(res)

        assert len(result_tabs) == 4
        assert len(titles) == 4
        assert titles == ["A", "B", "C", "D"]
        assert len(icons) == 4
        assert len(viewers) == 0

    def test_show_results(self, widget, res):
        """Tests QResultWidget show_results."""

        widget.show_results(res)

        assert len(widget._tab.children) == 4
        assert widget._tab.get_title(0) == "A"
        assert widget._tab.get_title(1) == "B"
        assert widget._tab.get_title(2) == "C"
        assert widget._tab.get_title(3) == "D"
        assert widget._tab.selected_index == 0
        assert len(widget._viewers) == 0

    def test_tab_index_changed(self, widget, res):
        """Tests QResultWidget _tab_index_changed."""

        assert len(widget._tab.children) == 0

        widget.show_results(res)
        assert len(widget._tab.children) == 4
        assert widget._tab.selected_index == 0

        # TODO check why below assertion fails
        #        assert widget._tab.children[0].loaded == True
        assert widget._tab.children[1].loaded == False
        assert widget._tab.children[2].loaded == False
        assert widget._tab.children[3].loaded == False

        widget._tab.selected_index = 1
        assert widget._tab.children[1].loaded == True

        widget._tab.selected_index = 2
        assert widget._tab.children[2].loaded == True

    def test_reset(self, widget, res):
        """Tests QResultWidget reset."""

        widget.show_results(res)
        widget.reset()

        assert widget._viewers is None
        assert widget._tab is not None
        assert len(widget._tab.children) == 0
        assert widget._tab.selected_index is None
        assert widget._tab.title_icons == []
        assert widget._tab._titles == {}


class TestResultTabPageWidget:
    """Tests ResultTabPageWidget."""

    @pytest.fixture
    def widget(self, res):

        return ResultTabPageWidget(title="A", nras=res["A"])

    # TODO test with a resultset that contains image
    # TODO test with a resultset that contains more rows than DOWNLOAD_THRESHOLD.

    def test_create(self, widget):
        """Tests ResultTabPageWidget constructor."""

        title = "A"
        assert widget._df.shape == (3, 2)
        assert widget._total_nb_rows == 3
        assert widget.loaded == False
        assert hasattr(widget, "_table") == False
        assert len(widget._cell_viewers) == 0
        assert isinstance(widget._columns_manager.get_column_feeder(0), ColumnFeeder)
        assert widget._columns_manager.hasVBRColumn == False
        # check if download link is created
        assert widget._hbox_title.children[0].children[1] is not None
        assert widget._hbox_title.children[0].children[1].filename == f"{title}.csv.gz"
        assert widget._hbox_title.children[0].children[1].mimetype == "application/gz"
        assert (
            widget._hbox_title.children[0].children[1].tooltip
            == f"Download {title}.csv.gz file."
        )
        # check tab title
        assert widget._hbox_title.children[0].children[0].value == f"<h3>{title}</h3>"

    def test_create_title(self, widget):
        """Tests ResultTabPageWidget _create_title."""

        title = "title"

        title_widget = widget._create_title(title, [])

        assert len(title_widget) == 2
        # title
        assert title_widget[0].children[0].value == f"<h3>{title}</h3>"
        assert title_widget[0].children[1].filename == f"{title}.csv.gz"
        assert title_widget[0].children[1].disabled == False
        assert title_widget[0].children[1].mimetype == "application/gz"
        assert title_widget[0].children[1].tooltip == f"Download {title}.csv.gz file."
        # paginator
        assert title_widget[0].children[2] is not None
        assert title_widget[0].children[2].layout.visibility == "hidden"

        # controls
        assert title_widget[1] is not None
        assert len(title_widget[1].children) == 0

    def test_create_load(self, widget):
        """Tests ResultTabPageWidget load."""

        assert widget.loaded == False
        assert hasattr(widget, "_table") == False

        widget.load()

        assert widget.loaded == True
        assert widget._table is not None
        assert widget._table.rows == 3
        assert widget._table.columns == 2

    def test_create_load_with_limit(self, widget):
        """Tests ResultTabPageWidget load with limit less than row count."""

        assert widget.loaded == False
        assert hasattr(widget, "_table") == False

        widget._limit = 2
        widget.load()

        assert widget.loaded == True
        assert widget._table is not None
        assert widget._table.rows == 2
        assert widget._table.columns == 2

    def test_load_table(self, widget):
        """Tests ResultTabPageWidget _load_table setting different page number and limit values."""

        assert widget._limit == PaginationWidget.DEFAULT_LIMIT

        widget._load_table(1, 3)
        assert widget._table is not None
        assert widget._table.rows == 3
        assert widget._table.columns == 2

        widget._load_table(1, 2)
        assert widget._table is not None
        assert widget._table.rows == 2
        assert widget._table.columns == 2

        widget._load_table(2, 2)
        assert widget._table is not None
        assert widget._table.rows == 1
        assert widget._table.columns == 2

        assert widget._limit == PaginationWidget.DEFAULT_LIMIT

    # TODO also test for invalid page numbers

    @pytest.mark.asyncio
    async def test_page_number_changed(self, widget):
        """Tests ResultTabPageWidget _page_number_changed."""

        assert widget._df.shape == (3, 2)
        assert widget._total_nb_rows == 3

        widget._limit = 2

        widget.load()

        widget._page_number_changed(change=dict(new=2))

        # wait for _page_number_changed to execute as it is debounced
        await asyncio.sleep(0.6)

        assert widget._table.rows == 1
        assert widget._table.columns == 2

        assert len(widget.children) == 2

    @pytest.mark.asyncio
    async def test_limit_changed(self, widget):
        """Tests ResultTabPageWidget _limit_changed."""

        assert widget._df.shape == (3, 2)
        assert widget._total_nb_rows == 3

        widget._limit_changed(change=dict(new=2))

        # wait for _limit_changed to execute as it is debounced
        await asyncio.sleep(0.6)

        assert widget._limit == 2
        assert widget._table is not None
        assert widget._table.rows == 2
        assert widget._table.columns == 2

        assert len(widget.children) == 2

    # TODO test with an example which has viewers

    def test_get_viewers(self, widget):
        """Tests ResultTabPageWidget get_viewers."""

        assert len(widget._cell_viewers) == 0
        assert len(widget.get_viewers()) == 0
