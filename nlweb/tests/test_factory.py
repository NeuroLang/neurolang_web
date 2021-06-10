from typing import Tuple

import matplotlib
import neurolang
import nlweb
import pytest
from nlweb.viewers.factory import ColumnFeederFactory, ColumnsManager, ViewerFactory
from traitlets import TraitError


class TestViewerFactory:
    """Tests ViewerFactory."""

    def test_create(self):
        """Tests ViewerFactory constructor."""

        widget = ViewerFactory()
        assert widget.get_region_viewer() is not None


class TestColumnFeederFactory:
    """Tests ColumnFeederFactory."""

    def test_create(self, mock_viewerfactory, mock_resulttabpage):
        """Tests ColumnFeederFactory constructor."""

        factory = ColumnFeederFactory()

        column_tfidf = factory.get_column(
            None, neurolang.frontend.neurosynth_utils.TfIDf, None
        )
        assert isinstance(column_tfidf, nlweb.viewers.column.TfIDfColumn)

        column_studyid = factory.get_column(
            None, neurolang.frontend.neurosynth_utils.StudyID, None
        )
        assert isinstance(column_studyid, nlweb.viewers.column.StudIdColumn)

        column_str = factory.get_column(None, str, None)
        assert isinstance(column_str, nlweb.viewers.column.ColumnFeeder)

        column_vbr = factory.get_column(
            mock_resulttabpage, neurolang.regions.ExplicitVBR, mock_viewerfactory
        )
        assert isinstance(column_vbr, nlweb.viewers.column.ExplicitVBRColumn)

        column_vbroverlay = factory.get_column(
            mock_resulttabpage, neurolang.regions.ExplicitVBROverlay, mock_viewerfactory
        )
        assert isinstance(
            column_vbroverlay, nlweb.viewers.column.ExplicitVBROverlayColumn
        )

        column_fig = factory.get_column(
            mock_resulttabpage, matplotlib.figure.Figure, mock_viewerfactory
        )
        assert isinstance(column_fig, nlweb.viewers.column.MpltFigureColumn)


class TestColumnManager:
    """Tests ColumnManager."""

    @pytest.fixture
    def widget(self, mock_resulttabpage, mock_viewerfactory):
        """Creates a widget to be used by the tests."""

        column_types = Tuple[
            neurolang.frontend.neurosynth_utils.TfIDf,
            neurolang.frontend.neurosynth_utils.StudyID,
            neurolang.regions.ExplicitVBR,
            neurolang.regions.ExplicitVBROverlay,
        ]
        return ColumnsManager(mock_resulttabpage, column_types, mock_viewerfactory)

    @pytest.fixture
    def widget_no_vbr(self, mock_resulttabpage, mock_viewerfactory):
        """Creates a widget to be used by the tests for non VBR column types"""
        column_types = Tuple[
            neurolang.frontend.neurosynth_utils.TfIDf,
            neurolang.frontend.neurosynth_utils.StudyID,
        ]
        return ColumnsManager(mock_resulttabpage, column_types, mock_viewerfactory)

    def test_create(self, widget):
        """Tests ColumnManager constructor."""

        assert len(widget.columns) == 4
        assert isinstance(widget.columns[0], nlweb.viewers.column.TfIDfColumn)
        assert isinstance(widget.columns[1], nlweb.viewers.column.StudIdColumn)
        assert isinstance(widget.columns[2], nlweb.viewers.column.ExplicitVBRColumn)
        assert isinstance(
            widget.columns[3], nlweb.viewers.column.ExplicitVBROverlayColumn
        )
        assert widget._hasVBRColumn == True

    def test_create_no_vbr(self, widget_no_vbr):
        """Tests ColumnManager constructor with no VBR type columns."""

        assert len(widget_no_vbr.columns) == 2
        assert isinstance(widget_no_vbr.columns[0], nlweb.viewers.column.TfIDfColumn)
        assert isinstance(widget_no_vbr.columns[1], nlweb.viewers.column.StudIdColumn)
        assert widget_no_vbr._hasVBRColumn == False

    def test_get_cell_widget_tfidf(self, widget):
        """Tests ColumnManager get_cell_widget for type TfIDf."""

        assert isinstance(
            widget.get_cell_widget(0, 0.25), nlweb.viewers.cell.TfIDfWidget
        )

        with pytest.raises(TraitError):
            widget.get_cell_widget(0, 1.5)

        with pytest.raises(TraitError):
            widget.get_cell_widget(0, -1)

    def test_get_cell_widget_studyid(self, widget):
        """Tests ColumnManager get_cell_widget for type StudyID."""

        assert isinstance(
            widget.get_cell_widget(1, 122334), nlweb.viewers.cell.StudyIdWidget
        )

    def test_get_cell_widget_vbr(self, widget, vbr):
        """Tests ColumnManager get_cell_widget for type ExplicitVBR."""

        assert isinstance(
            widget.get_cell_widget(2, vbr), nlweb.viewers.cell.ExplicitVBRCellWidget
        )

        assert isinstance(
            widget.get_cell_widget(2, None), nlweb.viewers.cell.LabelCellWidget
        )

    def test_get_cell_widget_vbr_overlay(self, widget, vbr_overlay):
        """Tests ColumnManager get_cell_widget for type ExplicitVBROverlay."""

        assert isinstance(
            widget.get_cell_widget(3, vbr_overlay),
            nlweb.viewers.cell.ExplicitVBROverlayCellWidget,
        )

        assert isinstance(
            widget.get_cell_widget(3, None), nlweb.viewers.cell.LabelCellWidget
        )

    def test_get_viewers(self, widget):
        """Tests ColumnManager get_viewers."""
        viewers = widget.get_viewers()

        assert len(viewers) == 1

    def test_get_viewers_no_vbr(self, widget_no_vbr):
        """Tests ColumnManager get_viewers where column manager contains no VBR type column."""
        viewers = widget_no_vbr.get_viewers()

        assert len(viewers) == 0

    def test_get_controls(self, widget):
        """Tests ColumnManager get_controls."""
        controls = widget.get_controls()

        assert len(controls) == 4

    def test_get_controls_no_vbr(self, widget_no_vbr):
        """Tests ColumnManager get_controls where column manager contains no VBR type column."""
        controls = widget_no_vbr.get_controls()

        assert len(controls) == 0

    def test_column_feeder(self, widget):
        """Tests ColumnManager get_column_feeder."""
        assert isinstance(widget.get_column_feeder(0), nlweb.viewers.column.TfIDfColumn)
        assert isinstance(
            widget.get_column_feeder(1), nlweb.viewers.column.StudIdColumn
        )
        assert isinstance(
            widget.get_column_feeder(2), nlweb.viewers.column.ExplicitVBRColumn
        )
        assert isinstance(
            widget.get_column_feeder(3), nlweb.viewers.column.ExplicitVBROverlayColumn
        )

    def test_hasVBRColumn(self, widget):
        """Tests ColumnManager hasVBRColumn."""
        assert widget._hasVBRColumn == True

    def test_hasVBRColumn_no_vbr(self, widget_no_vbr):
        """Tests ColumnManager hasVBRColumn where column manager contains no VBR type column."""
        assert widget_no_vbr.hasVBRColumn == False
