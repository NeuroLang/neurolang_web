from ipywidgets import Button, Layout

import neurolang

from nlweb.viewers.cell import (
    ExplicitVBRCellWidget,
    ExplicitVBROverlayCellWidget,
    LabelCellWidget,
    StudyIdWidget,
    TfIDfWidget,
)
from nlweb.viewers.factory import ViewerFactory


class ColumnFeeder:
    """Base class for a column feeder which works as a factory to create cell widgets and their corresponding controls and viewers of a specific type of column."""

    def __init__(self):
        self._viewer = None
        self._controls = []

    @property
    def viewer(self):
        """Returns the special viewer widget for this column.

        Returns
        -------
             the special viewer widget for this column, `None` if no special viewer is required.
        """
        return self._viewer

    @property
    def controls(self):
        """Returns list of widgets that are used to control the widgets of this column.

        Returns
        -------
        list

        """
        return self._controls

    def get_widget(self, obj):
        """Returns a Label widget for the specified `obj`.

        Returns
        -------
        ipywidgets.widgets.Label

        """
        return LabelCellWidget(str(obj))


class ExplicitVBRColumn(ColumnFeeder):
    __ICON_ON = "eye"
    __ICON_OFF = "eye-slash"
    __ICON_UNSELECT = "times"
    __TOOLTIP_ON = "Turn off selected regions. This will keep regions selected, but remove them from the viewer."
    __TOOLTIP_OFF = (
        "Turn on selected regions. This will add selected regions to the viewer."
    )

    def __init__(self, result_tab):
        super().__init__()
        self.result_tab = result_tab

        self._viewer = ViewerFactory.get_region_viewer()

        self._turn_on_off_btn = Button(
            tooltip=ExplicitVBRColumn.__TOOLTIP_ON,
            icon=ExplicitVBRColumn.__ICON_ON,
            layout=Layout(width="30px", padding_top="20px"),
        )
        self.result_tab.icon = self._turn_on_off_btn.icon

        self._turn_on_off_btn.on_click(self._on_turn_on_off_btn_clicked)
        self._controls.append(self._turn_on_off_btn)

        self._unselect_btn = Button(
            tooltip="Unselect all selected regions.",
            icon=ExplicitVBRColumn.__ICON_UNSELECT,
            layout=Layout(width="30px", padding_top="20px"),
        )
        self._unselect_btn.on_click(self._on_unselect_clicked)
        self._controls.append(self._unselect_btn)

        self._column_on = True

        self._evbr_widget_list = []

    def get_widget(self, obj):
        """"""
        if isinstance(obj, neurolang.regions.ExplicitVBR):
            e_widget = ExplicitVBRCellWidget(obj, self._viewer)
            self._evbr_widget_list.append(e_widget)
            return e_widget
        else:
            return LabelCellWidget(str(obj))

    def _selected_images(self):
        images = []
        for e_widget in self._evbr_widget_list:
            if e_widget.is_region_selected:
                images.append(e_widget.image)
        return images

    def _on_unselect_clicked(self, b):
        images = []
        for e_widget in self._evbr_widget_list:
            if e_widget.is_region_selected:
                images.append(e_widget.image)
                e_widget.undo_select()
        self._viewer.remove(images)

    def _on_turn_on_off_btn_clicked(self, b):
        images = []
        for e_widget in self._evbr_widget_list:
            e_widget.disable_region(self._column_on)
            if e_widget.is_region_selected:
                images.append(e_widget.image)

        if self._column_on:
            self._column_on = False
            self._turn_on_off_btn.icon = ExplicitVBRColumn.__ICON_OFF
            self._turn_on_off_btn.tooltip = ExplicitVBRColumn.__TOOLTIP_OFF
            self._unselect_btn.disabled = True
            self._viewer.remove(images)
        else:
            if self._viewer.can_add(images):
                self._viewer.add(images)
                self._column_on = True
                self._turn_on_off_btn.icon = ExplicitVBRColumn.__ICON_ON
                self._turn_on_off_btn.tooltip = ExplicitVBRColumn.__TOOLTIP_ON
                self._unselect_btn.disabled = False
            else:
                for e_widget in self._evbr_widget_list:
                    e_widget.disable_region(True)
                self._viewer.set_error(
                    "Papaya viewer does not allow more than 8 overlays. \nPlease unselect region to be able to add  new ones!"
                )

        self.result_tab.icon = self._turn_on_off_btn.icon


class ExplicitVBROverlayColumn(ExplicitVBRColumn):
    def __init__(self, result_tab):
        super().__init__(result_tab)

    def get_widget(self, obj):
        """"""
        if isinstance(obj, neurolang.regions.ExplicitVBROverlay):
            e_widget = ExplicitVBROverlayCellWidget(obj, self._viewer)
            self._evbr_widget_list.append(e_widget)
            return e_widget
        else:
            return LabelCellWidget(str(obj))


class StudIdColumn(ColumnFeeder):
    def __init__(self):
        super().__init__()

    def get_widget(self, obj):
        return StudyIdWidget(str(obj))


class TfIDfColumn(ColumnFeeder):
    def __init__(self):
        super().__init__()

    def get_widget(self, obj):
        return TfIDfWidget(float(obj))
