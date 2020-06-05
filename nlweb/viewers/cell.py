from functools import partial

from ipywidgets import (
    BoundedIntText,
    Button,
    Checkbox,
    Dropdown,
    FloatSlider,
    HBox,
    IntText,
    Label,
    Layout,
    VBox,
)

import neurolang

from nlweb.viewers import CellWidget, PapayaImage

from neurolang_ipywidgets import NlLink, NlProgress, NlCheckbox, NlPapayaViewer


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
        self._image = PapayaImage(obj.spatial_image())

        # default config for images
        self._image.config = dict(min=0, max=10, lut="Red Overlay")
        self._centered = False
        self._can_select = True

        self.layout.width = "160px"
        self.layout.flex_flow = "row"
        self.layout.display = "flex"

        # add widgets
        self._region_checkbox = NlCheckbox(
            value=False,
            description="show region",
            indent=False,
            layout=Layout(
                width="120px", margin="5px 15px 5px 0", padding="5px 15px 5px 15px"
            ),
        )
        self._center_btn = Button(
            tooltip="Center on region", icon="map-marker", layout=Layout(width="30px")
        )

        # add handlers
        self._region_checkbox.observe(
            partial(self._selection_changed, image=self._image), names="value"
        )

        self._center_btn.on_click(self._center_btn_clicked)

        self.children = [self._region_checkbox, self._center_btn]

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

        self.layout.width = "220px"

        self._image.config = {}

        self._config = PapayaConfigWidget(self)

        self._config_btn = Button(
            tooltip="Configure", icon="cog", layout=Layout(width="30px")
        )

        self._config_btn.on_click(self._config_btn_clicked)

        self.children = self.children + (self._config_btn,)

    def _config_btn_clicked(self, event):
        if self.children[-1] != self._config:
            self.layout.width = "600px"
            self.children = self.children + (self._config,)
        else:
            self.layout.width = "220px"
            self.children = self.children[:-1]

    def update_config(self):
        if self._region_checkbox.value:
            self._viewer.set_images()


class PapayaConfigWidget(VBox):
    """A widget that displays widgets to adjust NLPapayaViewer image parameters."""

    def __init__(self, parent: ExplicitVBROverlayCellWidget, *args, **kwargs):
        """
        Parameters
        ----------
        parent : ExplicitVBROverlayCellWidget
            parent widget that will display this widget.
        """
        super().__init__(*args, **kwargs)

        self._parent = parent
        self.layout.align_items = "center"

        # add widgets
        # sets alpha value
        self._alpha = FloatSlider(
            value=1,
            min=0,
            max=1.0,
            step=0.1,
            description="alpha:",
            description_tooltip="Overlay image alpha level (0 to 1).",
            disabled=False,
            continuous_update=False,
            orientation="horizontal",
            readout=True,
            readout_format=".1f",
            layout=Layout(width="200px"),
        )

        # sets lut value
        self._lut = Dropdown(
            options=[
                ("Grayscale", 1),
                ("Red Overlay", 2),
                ("Green Overlay", 3),
                ("Blue Overlay", 4),
                ("Gold", 5),
                ("Spectrum", 6),
                ("Overlay (Positives)", 7),
                ("Overlay (Negatives)", 8),
                ("Hot-and-Cold", 9),
            ],
            value=2,
            description="lut:",
            description_tooltip="The color table name.",
            layout=Layout(width="200px"),
        )

        # sets negative_lut value
        self._nlut = Dropdown(
            options=[
                ("Grayscale", 1),
                ("Red Overlay", 2),
                ("Green Overlay", 3),
                ("Blue Overlay", 4),
                ("Gold", 5),
                ("Spectrum", 6),
                ("Overlay (Positives)", 7),
                ("Overlay (Negatives)", 8),
                ("Hot-and-Cold", 9),
            ],
            value=2,
            description="negative-lut:",
            description_tooltip="The color table name used by the negative side of the parametric pair.",
            layout=Layout(width="10px"),
        )

        # sets min value
        self._min = IntText(
            value=None,
            description="min:",
            description_tooltip="The display range minimum.",
            disabled=False,
            layout=Layout(width="100px"),
        )

        # sets minPercent value
        self._minp = BoundedIntText(
            value=None,
            min=100,
            max=100,
            step=1,
            description="min %:",
            description_tooltip="The display range minimum as a percentage of image max.",
            disabled=False,
            layout=Layout(width="100px"),
        )

        # sets max value
        self._max = IntText(
            value=None,
            description="max:",
            description_tooltip="The display range maximum.",
            disabled=False,
            layout=Layout(width="100px"),
        )

        # sets maxPercent value
        self._maxp = BoundedIntText(
            value=None,
            min=100,
            max=100,
            step=1,
            description="max %:",
            description_tooltip="The display range minimum as a percentage of image max.",
            disabled=False,
            layout=Layout(width="100px"),
        )

        # sets symmetric value
        self._sym = Checkbox(
            value=False,
            description="symmetric",
            description_tooltip="When selected, sets the negative range of a parametric pair to the same size as the positive range.",
            disabled=False,
            #            indent=False,
            layout=Layout(width="50px"),
        )

        # add handlers
        self._alpha.observe(
            partial(self._config_changed, image=self._parent.image, name="alpha"),
            names="value",
        )

        self._lut.observe(
            partial(self._lut_changed, image=self._parent.image, name="lut"),
            names="value",
        )

        self._nlut.observe(
            partial(self._lut_changed, image=self._parent.image, name="negative_lut"),
            names="value",
        )

        self._min.observe(
            partial(self._config_changed, image=self._parent.image, name="min"),
            names="value",
        )

        self._minp.observe(
            partial(self._config_changed, image=self._parent.image, name="minPercent"),
            names="value",
        )

        self._max.observe(
            partial(self._config_changed, image=self._parent.image, name="max"),
            names="value",
        )

        self._maxp.observe(
            partial(self._config_changed, image=self._parent.image, name="maxPercent"),
            names="value",
        )

        self._sym.observe(
            partial(self._sym_changed, image=self._parent.image), names="value"
        )

        self.children = (
            self._alpha,
            self._lut,
            self._nlut,
            HBox([self._min, self._minp], layout=Layout(width="300px")),
            HBox([self._max, self._maxp], layout=Layout(width="300px")),
            self._sym,
        )

    def _config_changed(self, change, image, name):
        image.config[name] = change.new
        self._parent.update_config()

    def _lut_changed(self, change, image, name):
        image.config[name] = change.owner.label
        self._parent.update_config()

    def _sym_changed(self, change, image):
        """Updates `symmetric` config value when `_sym` checkbox is checked.

        Parameters
        ----------
        change:
            change in `_sym` checkbox.
        image: PapayaImage
        """
        value = "false"
        if change.new:
            value = "true"
        image.config["symmetric"] = value
        self._parent.update_config()
