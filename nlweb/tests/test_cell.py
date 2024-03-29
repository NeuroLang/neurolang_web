import matplotlib.pyplot as plt
import pytest
from ipywidgets import Widget
from traitlets import TraitError

from ..viewers.cell import (
    ExplicitVBRCellWidget,
    ExplicitVBROverlayCellWidget,
    LabelCellWidget,
    MpltFigureCellWidget,
    StudyIdWidget,
    TfIDfWidget,
)


class TestWidget:
    def teardown(self):
        for w in tuple(Widget.widgets.values()):
            w.close()


class TestStudyIdWidget(TestWidget):
    """Tests StudyIdWidget."""

    def test_create_no_params(self):
        """Tests constructor with no value specified for `study_id`."""
        with pytest.raises(TypeError):
            StudyIdWidget()

    def test_create_none_params(self):
        """Tests constructor with `None` specified for `study_id`."""
        with pytest.raises(TypeError):
            StudyIdWidget(study_id=None)

    def test_create_empty_string(self):
        """Tests constructor with empty string specified for `study_id`."""
        widget = StudyIdWidget(study_id="")
        assert widget.value == f"{StudyIdWidget._PubMed}:"
        assert widget.href == StudyIdWidget._URL

    def test_create(self):
        """Tests constructor with valid value specified for `study_id`."""
        id = "10908189"
        widget = StudyIdWidget(study_id=id)
        assert widget.value == f"{StudyIdWidget._PubMed}:{id}"
        assert widget.href == StudyIdWidget._URL + id


class TestTfIDfWidget(TestWidget):
    """Tests TfIDfWidget."""

    def test_create_no_params(self):
        """Tests constructor with no value specified for `tfidf`."""
        with pytest.raises(TypeError):
            TfIDfWidget()

    def test_create_none_params(self):
        """Tests constructor with `None` specified for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf=None)

    def test_create_str_params(self):
        """Tests constructor with type `str` specified for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf="")

    def test_create_negative_params(self):
        """Tests constructor with negative value specified for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf=-1)

    def test_create_invalid_params(self):
        """Tests constructor with value greater than 1 for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf=1.1)

    def test_create_valid_params(self):
        """Tests constructor with value greater than 1 for `tfidf`."""
        widget = TfIDfWidget(tfidf=0.1)
        assert widget.value == 0.1
        assert widget.max == 1


class TestLabelCellWidget(TestWidget):
    """Tests LabelCellWidget."""

    def test_create_no_params(self):
        """Tests constructor with no value specified for `value`."""
        widget = LabelCellWidget()
        assert widget.value == ""

    def test_create_emprty_str_params(self):
        """Tests constructor with empty string specified for `value`."""
        widget = LabelCellWidget("")
        assert widget.value == ""

    def test_create_none_params(self):
        """Tests constructor with `None` specified for `value`."""
        widget = LabelCellWidget(None)
        assert widget.value == ""

    def test_create(self):
        """Tests constructor with non empty string value specified for `value`."""
        test_label = "test_label"
        widget = LabelCellWidget(test_label)
        assert widget.value == test_label


class TestExplicitVBRCellWidget(TestWidget):
    """Tests ExplicitVBRCellWidget."""

    @pytest.fixture
    def widget(self, vbr, mock_viewer):
        return ExplicitVBRCellWidget(vbr, mock_viewer)

    def test_create_no_params(self):
        """Tests constructor with no parameters."""
        with pytest.raises(TypeError):
            ExplicitVBRCellWidget()

    def test_create_vbr_none(self, mock_viewer):
        """Tests constructor with `None` value for `vbr`."""

        with pytest.raises(TypeError) as error:
            ExplicitVBRCellWidget(vbr=None, viewer=mock_viewer)
        assert error.value.args[0] == "vbr should not be NoneType!"

    def test_create_viewer_none(self, vbr):
        """Tests constructor with `None` value for `viewer`."""

        with pytest.raises(TypeError) as error:
            ExplicitVBRCellWidget(vbr=vbr, viewer=None)
        assert error.value.args[0] == "viewer should not be NoneType!"

    def test_create(self, widget):
        """Tests constructor with valid values for `vbr` and `viewer`."""

        assert widget._viewer is not None
        assert widget.image is not None
        assert widget.is_region_selected == False
        assert widget._download_link is not None
        assert widget._download_link.filename == f"{widget.image.id}.nii.gz"
        assert widget._center_btn is not None
        assert widget._center_btn.icon == "map-marker"
        assert widget._centered == False
        assert widget._can_select == True

    def test_disable_region(self, widget):
        """Tests `disable_region`."""

        widget.disable_region(True)
        assert widget._region_checkbox.disabled == True
        assert widget._center_btn.disabled == True

        widget.disable_region(False)
        assert widget._region_checkbox.disabled == False
        assert widget._center_btn.disabled == False

    def test_center_region(self, widget):
        """Tests `center_region`."""

        widget.center_region(True)
        assert widget._centered == True
        assert widget._center_btn.icon == "map-pin"

        widget.center_region(False)
        assert widget._centered == False
        assert widget._center_btn.icon == "map-marker"

    def test_remove_center(self, widget):
        """Tests `remove_center`."""

        widget.remove_center()
        assert widget._centered == False
        assert widget._center_btn.icon == "map-marker"


class TestExplicitVBROverlayCellWidget(TestExplicitVBRCellWidget):
    """Tests ExplicitOverlayVBRCellWidget."""

    @pytest.fixture
    def widget(self, vbr_overlay, mock_viewer):
        return ExplicitVBROverlayCellWidget(vbr_overlay, mock_viewer)

    def test_create_no_params(self):
        """Tests constructor with no parameters."""
        with pytest.raises(TypeError):
            ExplicitVBROverlayCellWidget()

    def test_create_vbr_none(self, mock_viewer):
        """Tests constructor with `None` value for `vbr`."""

        with pytest.raises(TypeError) as error:
            ExplicitVBROverlayCellWidget(vbr=None, viewer=mock_viewer)
        assert error.value.args[0] == "vbr should not be NoneType!"

    def test_create_viewer_none(self, vbr_overlay):
        """Tests constructor with `None` value for `viewer`."""

        with pytest.raises(TypeError) as error:
            ExplicitVBROverlayCellWidget(vbr=vbr_overlay, viewer=None)
        assert error.value.args[0] == "viewer should not be NoneType!"


class TestMpltFigureCellWidget(TestWidget):
    """Tests MpltFigureCellWidget."""

    @pytest.fixture
    def fig(self):
        return plt.figure()

    @pytest.fixture
    def widget(self, fig, mock_figure_viewer):
        return MpltFigureCellWidget(fig, mock_figure_viewer)

    def test_create_no_params(self):
        """Tests constructor with no parameters."""
        with pytest.raises(TypeError):
            MpltFigureCellWidget()

    def test_create_fig_none(self, mock_figure_viewer):
        """Tests constructor with `None` value for `fig`."""

        with pytest.raises(TypeError) as error:
            MpltFigureCellWidget(fig=None, viewer=mock_figure_viewer)
        assert error.value.args[0] == "fig should not be None!"

    def test_create_viewer_none(self, fig):
        """Tests constructor with `None` value for `viewer`."""

        with pytest.raises(TypeError) as error:
            MpltFigureCellWidget(fig=fig, viewer=None)
        assert error.value.args[0] == "viewer should not be None!"

    def test_create(self, widget):
        """Tests constructor with valid values for `fig` and `viewer`."""

        assert widget._viewer is not None
        assert widget.fig is not None
        assert widget.is_figure_selected == False
        assert widget._show_fig_checkbox is not None

    def test_unselect_figure(self, widget):
        """Tests `unselect_figure`."""

        widget.unselect_figure()
        assert not widget.is_figure_selected
        widget._viewer.reset.assert_not_called()

        widget._show_fig_checkbox.value = True
        widget.unselect_figure()
        assert not widget.is_figure_selected
        widget._viewer.reset.assert_not_called()

    def test_select_figure(self, widget):
        widget._show_fig_checkbox.value = True
        assert widget.is_figure_selected
        widget._viewer.show_figure.assert_called_with(widget.fig, widget)
