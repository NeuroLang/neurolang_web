from functools import partial

from ipywidgets import Button, HBox, Label, Layout

import neurolang

from nlweb.viewers import CellWidget, PapayaViewerWidget

from neurolang_ipywidgets import NlLink, NlProgress, NlCheckbox


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
    """ A cell widget for data type `ExplicitVBR` that displays a checkbox connected to a viewer that visualizes spatial image of `ExplicitVBR`.
    """

    def __init__(
        self,
        obj: neurolang.regions.ExplicitVBR,
        viewer: PapayaViewerWidget,
        *args,
        **kwargs,
    ):
        """Initializes the widget with the specified `obj`.

        Parameters
        ----------
        obj: neurolang.regions.ExplicitVBR

        viewer : PapayaViewerWidget

        """
        super().__init__(*args, **kwargs)

        # viewer that visualizes the spatial image when checkbox is checked.
        self._viewer = viewer

        self._can_select = True

        self.__image = obj.spatial_image()

        self._region_checkbox = NlCheckbox(
            value=False,
            description="show region",
            layout=Layout(
                width="120px", margin="5px 15px 5px 0", padding="5px 15px 5px 15px"
            ),
        )
        self._region_checkbox.observe(
            partial(self._selection_changed, image=self.__image), names="value"
        )

        self._center_btn = Button(
            tooltip="Center on region", icon="map-marker", layout=Layout(width="30px")
        )
        self._center_btn.on_click(self._center_btn_clicked)
        self._centered = False

        self.layout.align_items = "center"

        self.children = [
            self._region_checkbox,
            self._center_btn,
        ]

    @property
    def image(self):
        return self.__image

    @property
    def is_region_selected(self):
        return self._region_checkbox.value

    def disable_region(self, is_disabled):
        self._region_checkbox.disabled = is_disabled
        self._center_btn.disabled = is_disabled

    def unselect_region(self):
        self._region_checkbox.value = False

    def unselect_region_without_remove(self):
        self._can_select = False
        self._region_checkbox.value = False
        self._can_select = True

    def _selection_changed(self, change, image):
        if self._can_select:
            if change["new"]:
                self._viewer.add(self, [image])
            else:
                self._viewer.remove(self, [image])

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
