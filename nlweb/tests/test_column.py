import pytest

from traitlets import TraitError

from nlweb.viewers.column import (
    ColumnFeeder,
    ExplicitVBRColumn,
    ExplicitVBROverlayColumn,
    StudIdColumn,
    TfIDfColumn,
)
import nlweb


class TestColumnFeeder:
    """Tests ColumnFeeder."""

    @pytest.fixture
    def column(self):
        return ColumnFeeder()

    def test_create(self, column):
        """Tests ColumnFeeder constructor."""

        assert column._viewer is None
        assert column._controls == []

    def test_viewer(self, column):
        """Tests ColumnFeeder viewer property."""
        assert column.viewer is None

    def test_controls(self, column):
        """Tests ColumnFeeder controls property."""
        assert column.controls == []

    def test_get_widget(self, column):
        """Tests ColumnFeeder get_widget."""
        assert column.get_widget(25) == "25"
        assert column.get_widget(25.5) == "25.5"
        assert column.get_widget(None) == "None"


class TestExplicitVBRColumn:
    """Tests ExplicitVBRColumn."""

    @pytest.fixture
    def column(self, mock_resulttabpage, mock_viewerfactory):
        return ExplicitVBRColumn(mock_resulttabpage, mock_viewerfactory)

    def test_create(self, column, mock_resulttabpage, mock_viewerfactory):
        """Tests ExplicitVBRColumn constructor."""

        assert column.result_tab == mock_resulttabpage
        assert column.viewer == mock_viewerfactory.get_region_viewer()
        assert column._turn_on_off_btn is not None
        assert len(column._controls) == 2
        assert column._column_on == True
        assert column._evbr_widget_list == []

    def test_get_widget(self, column, vbr):
        """Tests ExplicitVBRColumn get_widget."""
        assert isinstance(
            column.get_widget(vbr), nlweb.viewers.cell.ExplicitVBRCellWidget
        )

        assert isinstance(column.get_widget(None), nlweb.viewers.cell.LabelCellWidget)


class TestExplicitVBROverlayColumn:
    """Tests ExplicitVBROverlayColumn."""

    @pytest.fixture
    def column(self, mock_resulttabpage, mock_viewerfactory):
        return ExplicitVBROverlayColumn(mock_resulttabpage, mock_viewerfactory)

    def test_get_widget(self, column, vbr_overlay):
        """Tests ExplicitVBROverlayColumn get_widget."""
        assert isinstance(
            column.get_widget(vbr_overlay),
            nlweb.viewers.cell.ExplicitVBROverlayCellWidget,
        )

        assert isinstance(column.get_widget(None), nlweb.viewers.cell.LabelCellWidget)


class TestStudyIdColumn:
    """Tests StudyIdColumn."""

    @pytest.fixture
    def column(self):
        return StudIdColumn()

    def test_get_widget(self, column):
        """Tests StudIdColumn get_widget."""
        assert isinstance(column.get_widget(123423), nlweb.viewers.cell.StudyIdWidget)


class TestTfIDfColumn(TestColumnFeeder):
    """Tests TfIDfColumn."""

    @pytest.fixture
    def column(self):
        return TfIDfColumn()

    def test_get_widget(self, column):
        """Tests TfIDfColumn get_widget."""
        assert isinstance(column.get_widget(0.25), nlweb.viewers.cell.TfIDfWidget)

        with pytest.raises(TraitError):
            column.get_widget(1.5)

        with pytest.raises(TraitError):
            column.get_widget(-1)
