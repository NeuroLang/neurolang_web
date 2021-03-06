from ipywidgets import Layout

import neurolang
import typing

from .papaya_widget import PapayaWidget


class ViewerFactory:
    def __init__(self):
        self.__papaya_viewer = PapayaWidget(
            layout=Layout(
                width="auto",
                height="580px",
                max_width="950px",
                max_height="780px",
                overflow="hidden",
                overflow_y="hidden",
            )
        )

    def get_region_viewer(self):
        return self.__papaya_viewer


class ColumnFeederFactory:
    """A factory class that creates `ColumnFeeder`s for specified column types."""

    @staticmethod
    def get_column(result_tab, column_type, viewer_factory):
        """Creates and returns a `ColumnFeeder` for the specified `column_type`.

        Parameters
        ----------
        result_tab: ResultTabWidget
            the result tab that views the required column.
        column_type: str
            type of the column for the required `ColumnFeeder`.
        viewer_factory: ViewerFactory
            viewer factory to get viewer for corresponding column type


        Returns
        -------
        ColumnFeeder
            column feeder for the specified `column_type`.

        """
        import nlweb.viewers.column

        if column_type == neurolang.regions.ExplicitVBR:
            return nlweb.viewers.column.ExplicitVBRColumn(result_tab, viewer_factory)
        elif column_type == neurolang.regions.ExplicitVBROverlay:
            return nlweb.viewers.column.ExplicitVBROverlayColumn(
                result_tab, viewer_factory
            )
        elif column_type == neurolang.frontend.neurosynth_utils.StudyID:
            return nlweb.viewers.column.StudIdColumn()
        elif column_type == neurolang.frontend.neurosynth_utils.TfIDf:
            return nlweb.viewers.column.TfIDfColumn()
        else:
            return nlweb.viewers.column.ColumnFeeder()


class ColumnsManager:
    """A class that creates column feeders for a specified `tuple` of column types and manages creation of widgets for each column and, their corresponding viewers and controls. """

    def __init__(
        self, result_tab, column_types: typing.Tuple, viewer_factory: ViewerFactory
    ):
        """
        Parameters
        ----------
        result_tab: ResultTabPageWidget
            the tab widget that will display the columns generated by this column manager.
        column_types: tuple
            tuple that contains the column types to be generated.
        viewer_factory: ViewerFactory
            viewer factory to get viewer for corresponding column type.
        """
        self.columns = []

        self._hasVBRColumn = False

        for column_type in column_types.__args__:
            if (
                column_type == neurolang.regions.ExplicitVBR
                or column_type == neurolang.regions.ExplicitVBROverlay
            ):
                self._hasVBRColumn = True
            self.columns.append(
                ColumnFeederFactory.get_column(result_tab, column_type, viewer_factory)
            )

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

    def get_column_feeder(self, index):
        """Returns column feeder at the specified `index`.

        Parameters
        ----------
        index: int
            index of column feeder.

        Returns
        -------
        ColumnFeeder
            the column feeder at the specified `index`.
        """
        return self.columns[index]

    @property
    def hasVBRColumn(self):
        """Returns `True` if this column manager contains a `neurolang.regions.ExplicitVBR` or `neurolang.regions.ExplicitVBROverlay` column; False otherwise. """
        return self._hasVBRColumn
