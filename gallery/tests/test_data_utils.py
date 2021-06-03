import os
from neurolang.frontend.neurosynth_utils import StudyID

import pandas as pd
from pathlib import Path
from gallery.data_utils import (
    fetch_neuroquery,
    read_and_convert_csv_to_hdf,
)
from pandas.testing import assert_frame_equal


def test_read_and_convert_csv_to_hdf(tmp_path):
    """
    Test that read_and_convert_csv_to_hdf creates an hdf file and then reads
    from this hdf file on subsequent calls
    Parameters
    ----------
    tmp_path : Path
        fixture which will provide a temporary directory unique to 
        the test invocation, created in the base temporary directory.
    """
    # Create csv file to read
    df = pd.DataFrame({"study_id": [123, 456], "term": ["brain", "memory"]})
    file = tmp_path / "test.txt"
    df.to_csv(file, sep="\t", index=False)

    # Read and convert csv_file
    res = read_and_convert_csv_to_hdf(file, sep="\t")

    assert os.path.exists(tmp_path / "test.h5")
    assert_frame_equal(df, res)

    # Delete csv_file
    os.remove(file)
    res = read_and_convert_csv_to_hdf(file, sep="\t")
    assert_frame_equal(df, res)


def test_fetch_neuroquery():
    """
    Function should load the fixture files in the neuroquery data dir
    and return dataframes with proper shape and types.
    """
    data_dir = Path(os.path.dirname(os.path.realpath(__file__))) / "data"
    term_data, peak_data, study_ids = fetch_neuroquery(None, data_dir)

    assert all(term_data.columns == ["term", "tfidf", "study_id"])
    assert isinstance(term_data["study_id"].values[0], StudyID)
    assert term_data.shape == (14, 3)
    assert peak_data.shape == (20, 4)
    assert isinstance(peak_data["study_id"].values[0], StudyID)
    assert study_ids.shape == (20, 1)
    assert isinstance(study_ids["study_id"].values[0], StudyID)
