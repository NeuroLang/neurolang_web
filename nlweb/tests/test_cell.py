import pytest
from ipywidgets import Widget

from ..viewers.cell import StudyIdWidget


class TestStudyIdWidget:
    def teardown(self):
        for w in tuple(Widget.widgets.values()):
            w.close()

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
        assert widget.value == StudyIdWidget._PubMed + ":"
        assert widget.href == StudyIdWidget._URL
