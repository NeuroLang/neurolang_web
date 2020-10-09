from neurolang.frontend import ExplicitVBR, ExplicitVBROverlay, NeurolangDL
import pytest
import numpy as np
from os.path import dirname, join


DATA_DIR = join(dirname(__file__), "data")
VBR_VOXELS = "vbr_voxels.npz"
VBR_AFFINE = "vbr_affine.npy"


@pytest.fixture
def vbr(monkeypatch):

    voxels = np.transpose(np.load(join(DATA_DIR, VBR_VOXELS))["arr_0"].nonzero())
    affine = np.load(join(DATA_DIR, VBR_AFFINE))

    return ExplicitVBR(voxels, affine, image_dim=(91, 109, 91), prebuild_tree=True)


@pytest.fixture
def vbr_overlay(monkeypatch):
    def randint(size=None):
        return np.random.randint(0, 256, size=size)

    voxels = np.transpose(np.load(join(DATA_DIR, VBR_VOXELS))["arr_0"].nonzero())
    affine = np.load(join(DATA_DIR, VBR_AFFINE))

    overlay = randint(size=voxels.shape[0])

    return ExplicitVBROverlay(
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
    return MockNlPapayaViewer()


@pytest.fixture
def engine(monkeypatch):
    return NeurolangDL()


@pytest.fixture
def res(monkeypatch):
    neurolang = NeurolangDL()
    neurolang.execute_datalog_program(
        """
    A(4, 5)
    A(5, 6)
    A(6, 5)
    B(x,y) :- A(x, y)
    B(x,y) :- B(x, z),A(z, y)
    C(x) :- B(x, y), y == 5
    D("x")
    """
    )
    return neurolang.solve_all()


@pytest.fixture
def mock_solve_all(monkeypatch, res):
    def solve_all(*args, **kwargs):
        return res

    return solve_all
