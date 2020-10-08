import pytest

from ..viewers.query import QueryWidget


class TestQueryWidget:
    """Tests QueryWidget."""

    @pytest.fixture
    def widget(self, engine):
        yield QueryWidget(neurolang_engine=engine)

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
        """Tests constructor with no value specified for `default_query`."""

        res = widget.run_query(widget.query.text)

        assert res == {}
