from ipywidgets import Layout

import neurolang
from neurolang_ipywidgets import NlPapayaViewer


class ViewerFactory:
    papaya_viewer = NlPapayaViewer(
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
        import nlweb.viewers.column

        if column_type == neurolang.regions.ExplicitVBR:
            return nlweb.viewers.column.ExplicitVBRColumn(result_tab)
        elif column_type == neurolang.regions.ExplicitVBROverlay:
            return nlweb.viewers.column.ExplicitVBROverlayColumn(result_tab)
        elif column_type == neurolang.frontend.neurosynth_utils.StudyID:
            return nlweb.viewers.column.StudIdColumn()
        elif (
            column_type == neurolang.frontend.neurosynth_utils.TfIDf
            or column_type == float
        ):
            return nlweb.viewers.column.TfIDfColumn()
        else:
            return nlweb.viewers.column.ColumnFeeder()


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
