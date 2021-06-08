from pathlib import Path
from nlweb.data_utils import (
    fetch_difumo,
    fetch_neuroquery,
    fetch_neurosynth,
    load_mni_atlas,
)


class FetchNeuroQuerySuite:
    """
    A benchmark that times the performance of the fetch_neuroquery function from
    data_utils.
    The self.data_dir attribute should point to a directory where the neuroquery
    database has already been downloaded, otherwise the benchmark will measure the time
    to download.
    """

    params = [[None, 1e-2], ["xyz", "ijk"], [False, True]]
    param_names = ["tfidf_threshold", "coord_type", "convert_study_ids"]

    def setup(self, tfidf_threshold, coord_type, convert_study_ids):
        self.data_dir = Path(__file__).parent.parent / "gallery" / "neurolang_data"
        self.resolution = 3
        self.mask = load_mni_atlas(data_dir=self.data_dir, resolution=self.resolution)

    def time_fetch_neuroquery(self, tfidf_threshold, coord_type, convert_study_ids):
        fetch_neuroquery(
            self.mask,
            data_dir=self.data_dir,
            tfidf_threshold=tfidf_threshold,
            coord_type=coord_type,
            convert_study_ids=convert_study_ids,
        )


class FetchDifumoSuite:
    """
    A benchmark that times the performance of the fetch_difumo function from
    data_utils.
    The self.data_dir attribute should point to a directory where the difumo
    database has already been downloaded, otherwise the benchmark will measure the time
    to download.
    """

    params = [
        [128, 256],
        ["xyz", "ijk"],
    ]
    param_names = ["n_components", "coord_type"]

    def setup(self, n_components, coord_type):
        self.data_dir = Path(__file__).parent.parent / "gallery" / "neurolang_data"
        self.resolution = 3
        self.mask = load_mni_atlas(data_dir=self.data_dir, resolution=self.resolution)

    def time_fetch_neuroquery(self, n_components, coord_type):
        fetch_difumo(
            self.mask,
            data_dir=self.data_dir,
            n_components=n_components,
            coord_type=coord_type,
        )


class FetchNeuroSynthSuite:
    """
    A benchmark that times the performance of the fetch_neurosynth function from
    data_utils.
    The self.data_dir attribute should point to a directory where the neurosynth
    database has already been downloaded, otherwise the benchmark will measure the time
    to download.
    """

    params = [[None, 1e-2], [False, True]]
    param_names = ["tfidf_threshold", "convert_study_ids"]

    def setup(self, tfidf_threshold, convert_study_ids):
        self.data_dir = Path(__file__).parent.parent / "gallery" / "neurolang_data"

    def time_fetch_neurosynth(self, tfidf_threshold, convert_study_ids):
        fetch_neurosynth(
            data_dir=self.data_dir,
            tfidf_threshold=tfidf_threshold,
            convert_study_ids=convert_study_ids,
        )
