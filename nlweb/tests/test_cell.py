import pytest
from ipywidgets import Widget

from traitlets import TraitError

from ..viewers.cell import StudyIdWidget, TfIDfWidget


class TestWidget:
    def teardown(self):
        for w in tuple(Widget.widgets.values()):
            w.close()


class TestStudyIdWidget(TestWidget):
    def test_create_no_params(self):
        """Tests StudyIdWidget constructor with no value specified for `study_id`."""
        with pytest.raises(TypeError):
            StudyIdWidget()

    def test_create_none_params(self):
        """Tests StudyIdWidget constructor with `None` specified for `study_id`."""
        with pytest.raises(TypeError):
            StudyIdWidget(study_id=None)

    def test_create_empty_string(self):
        """Tests StudyIdWidget constructor with empty string specified for `study_id`."""
        widget = StudyIdWidget(study_id="")
        assert widget.value == f"{StudyIdWidget._PubMed}:"
        assert widget.href == StudyIdWidget._URL

    def test_create(self):
        """Tests StudyIdWidget constructor with valid value specified for `study_id`."""
        id = "10908189"
        widget = StudyIdWidget(study_id=id)
        assert widget.value == f"{StudyIdWidget._PubMed}:{id}"
        assert widget.href == StudyIdWidget._URL + id


class TestTfIDfWidget(TestWidget):
    def test_create_no_params(self):
        """Tests TfIDfWidget constructor with no value specified for `tfidf`."""
        with pytest.raises(TypeError):
            TfIDfWidget()

    def test_create_none_params(self):
        """Tests TfIDfWidget constructor with `None` specified for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf=None)

    def test_create_str_params(self):
        """Tests TfIDfWidget constructor with type `str` specified for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf="")

    def test_create_negative_params(self):
        """Tests TfIDfWidget constructor with negative value specified for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf=-1)

    def test_create_invalid_params(self):
        """Tests TfIDfWidget constructor with value greater than 1 for `tfidf`."""
        with pytest.raises(TraitError):
            TfIDfWidget(tfidf=1.1)
