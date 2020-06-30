from functools import partial

from ipywidgets import Button, HBox, Label, Layout

import neurolang

from neurolang_ipywidgets import (
    NlLink,
    NlProgress,
    NlCheckbox,
    NlPapayaViewer,
    PapayaSpatialImage,
)

from nlweb.viewers import CellWidget


class StudyIdWidget(NlLink, CellWidget):
    """A widget to display PubMed study IDs as links to publications."""

    __URL = "https://www.ncbi.nlm.nih.gov/pubmed/?term="
    __PubMed = "PubMed"

    def __init__(self, study_id, *args, **kwargs):
        """
        Parameters
        ----------
        study_id : str, StudyID
            PubMed study ID.
        """
        super().__init__(
            value=StudyIdWidget.__PubMed + ":" + study_id,
            href=StudyIdWidget.__URL + study_id,
            *args,
            **kwargs,
        )


class TfIDfWidget(NlProgress, CellWidget):
    """A widget to display TfIDf value ."""

    def __init__(self, tfidf, *args, **kwargs):
        """
        Parameters
        ----------
        tfidf : float, TfIDf
            .
        """
        super().__init__(value=tfidf, max=1, *args, **kwargs)


class LabelCellWidget(Label, CellWidget):
    """A cell widget for data type `str` that simply displays the given string.

    Requires no additional viewer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ExplicitVBRCellWidget(HBox, CellWidget):
    """A cell widget for data type `ExplicitVBR` that enables displaying the spatial image in an associated viewer or center on the spatial image's coordinates.
    """

    def __init__(
        self,
        obj: neurolang.regions.ExplicitVBR,
        viewer: NlPapayaViewer,
        *args,
        **kwargs,
    ):
        """Initializes the widget with the specified `obj`.

        Parameters
        ----------
        obj: neurolang.regions.ExplicitVBR

        viewer : NlPapayaViewer
            associated viewer to visualize the spatial image.
        """
        super().__init__(*args, **kwargs)

        self._viewer = viewer
        self._image = PapayaSpatialImage(obj.spatial_image())
        # default config for images
        self._image.config = dict(
            min=0,
            max=10,
            lut="Red Overlay",
            symmetric="false",
            minPercent=100,
            maxPercent=100,
        )

        self._centered = False
        self._can_select = True

        # adjust layout
        self.layout.justify_content = "flex-start"
        self.layout.width = "160px"
        self.layout.display = "flex"
        self.layout.flex_direction = "row"
        self.layout.flex = "0 0 auto"

        self._init_widgets(self._image)

        self.children = [self._region_checkbox, self._center_btn]

    def _init_widgets(self, image):
        # add widgets
        self._region_checkbox = NlCheckbox(
            value=False,
            description="show region",
            indent=False,
            layout=Layout(
                width="120px",
                max_width="120px",
                min_width="120px",
                margin="5px 5px 5px 0",
                padding="0px 0px 0px 0px",
                flex="0 0 auto",
                align_self="flex-start",
            ),
        )
        self._center_btn = Button(
            tooltip="Center on region",
            icon="map-marker",
            layout=Layout(
                width="30px",
                max_width="30px",
                min_width="30px",
                margin="5px 5px 5px 0",
                padding="0 0 0 0",
                flex="0 0 auto",
                align_self="flex-start",
            ),
        )

        # add handlers
        self._region_checkbox.observe(
            partial(self._selection_changed, image=image), names="value"
        )

        self._center_btn.on_click(self._center_btn_clicked)

    @property
    def image(self):
        return self._image

    @property
    def is_region_selected(self):
        return self._region_checkbox.value

    def disable_region(self, is_disabled):
        self._region_checkbox.disabled = is_disabled
        self._center_btn.disabled = is_disabled

    def unselect_region(self):
        self._region_checkbox.value = False

    def undo_select(self):
        self._can_select = False
        self._region_checkbox.value = False
        self._can_select = True

    def _selection_changed(self, change, image):
        if self._can_select:
            if change["new"]:
                if self._viewer.can_add([image]):
                    self._viewer.add([image])
                else:
                    self.undo_select()
                    self._viewer.set_error(
                        "Papaya viewer does not allow more than 8 overlays. \nPlease unselect region to be able to add  new ones!"
                    )
            else:
                self._viewer.remove([image])

    def center_region(self, is_centered):
        self._centered = is_centered
        if is_centered:
            self._center_btn.icon = "map-pin"
        else:
            self._center_btn.icon = "map-marker"

    def _center_btn_clicked(self, b):
        if not self._centered:
            self.center_region(True)
            if not self._region_checkbox.value:
                self._region_checkbox.value = True
            self._viewer.set_center(self, self.image)

    def remove_center(self):
        self.center_region(False)


class ExplicitVBROverlayCellWidget(ExplicitVBRCellWidget):
    """ A cell widget for data type `ExplicitVBROverlay` that enables displaying the spatial image in an associated viewer or center on the spatial image's coordinates, and manipulate image configuration parameters.
    """

    def __init__(
        self,
        obj: neurolang.regions.ExplicitVBROverlay,
        viewer: NlPapayaViewer,
        *args,
        **kwargs,
    ):
        """Initializes the widget with the specified `obj`.

        Parameters
        ----------
        obj: neurolang.regions.ExplicitVBROverlay

        viewer : NlPapayaViewer
            associated viewer to visualize the spatial image.
        """
        super().__init__(obj, viewer, *args, **kwargs)

        self._image.config["max"] = 0.1

        self.layout.width = "200px"

        self._colorbar_btn = Button(
            tooltip="Show color bar",
            icon="tint",
            layout=self._center_btn.layout,
            disabled=True,
        )

        self._config_btn = Button(
            tooltip="Configure",
            icon="cog",
            layout=self._center_btn.layout,
            disabled=True,
        )

        self._config_btn.on_click(self._config_btn_clicked)
        self._colorbar_btn.on_click(self._colorbar_btn_clicked)

        self.children = self.children + (self._colorbar_btn, self._config_btn)

    def _config_btn_clicked(self, event):
        if self._config_btn.button_style == "":
            self._config_btn.button_style = "warning"
            self._viewer.show_image_config(self.image, True)
            self._viewer.observe(self._reset_config, names=["current_config"])
        else:
            self._reset_config(None)
            self._viewer.show_image_config(self.image, False)

    def _colorbar_btn_clicked(self, event):
        if self._colorbar_btn.button_style == "":
            self._colorbar_btn.button_style = "warning"
            self._viewer.show_image_colorbar(self.image)
            self._viewer.observe(self._reset_colorbar, names=["current_colorbar"])

    def _selection_changed(self, change, image):
        super()._selection_changed(change, image)
        self._config_btn.disabled = not self._region_checkbox.value
        self._colorbar_btn.disabled = not self._region_checkbox.value

        if not change["new"]:
            self._reset_config(None)
            self._reset_colorbar(None)
            self._viewer.show_image_config(self.image, False)
        else:
            self._colorbar_btn_clicked(None)

    def _reset_config(self, change):
        self._config_btn.button_style = ""
        try:
            self._viewer.unobserve(self._reset_config, names=["current_config"])
        except ValueError:
            pass

    def _reset_colorbar(self, change):
        self._colorbar_btn.button_style = ""
        try:
            self._viewer.unobserve(self._reset_colorbar, names=["current_colorbar"])
        except ValueError:
            pass
