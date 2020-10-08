from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay
import pytest
import numpy as np
from os.path import dirname, join


DATA_DIR = join(dirname(__file__), "data")
VBR_VOXELS = "vbr_voxels.npz"
VBR_AFFINE = "vbr_affine.npy"


@pytest.fixture
def mock_vbr(monkeypatch):

    voxels = np.transpose(np.load(join(DATA_DIR, VBR_VOXELS))["arr_0"].nonzero())
    affine = np.load(join(DATA_DIR, VBR_AFFINE))

    yield ExplicitVBR(voxels, affine, image_dim=(91, 109, 91), prebuild_tree=True)


@pytest.fixture
def mock_vbr_overlay(monkeypatch):
    def randint(size=None):
        return np.random.randint(0, 256, size=size)

    voxels = np.transpose(np.load(join(DATA_DIR, VBR_VOXELS))["arr_0"].nonzero())
    affine = np.load(join(DATA_DIR, VBR_AFFINE))

    overlay = randint(size=voxels.shape[0])

    yield ExplicitVBROverlay(
        voxels, affine, image_dim=(91, 109, 91), overlay=overlay, prebuild_tree=True
    )


class MockNlPapayaViewer:
    """A mock class for neurolang_ipywidgets.NlPapayaViewer"""

    def __init__(self, *args):
        pass

    def observe(self, *args, **kwargs):
        pass


@pytest.fixture
def mock_viewer(monkeypatch):
    yield MockNlPapayaViewer()
