import pytest

from neurolang.frontend import NeurolangDL

from ..viewers.query import QueryWidget


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

    def test_reset_output(self, widget):
        """Tests _reset_output."""

        widget._reset_output()
        assert widget.result_viewer.layout.visibility == "hidden"
        assert widget.error_display.layout.visibility == "hidden"

    def test_handle_generic_error(self, widget):
        """Tests _handle_generic_error."""
        widget._handle_generic_error(ValueError("Error"))
        assert (
            widget.error_display.value
            == f"<pre style='background-color:#faaba5; border: 1px solid red; padding: 0.4em'>{ValueError('Error')}</pre>"
        )
