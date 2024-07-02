"""Tests for generat module."""

import numpy as np
import pytest

from nima import generat


def test_bias() -> None:
    """Test generation of bias."""
    bias = generat.gen_bias(nrows=3, ncols=128)
    z = 2 - (100**2 / 2.6 * ((np.sin(100 / 20)) ** 2 + 0.1)) / 4000
    assert bias[1, 100] == z * 2


def test_flat() -> None:
    """Test generation of bias."""
    flat = generat.gen_flat(nrows=10, ncols=10)
    assert np.mean(flat) == 1.0


def test_flat_shape() -> None:
    """Test nrows and ncols."""
    obj = generat.gen_flat(nrows=12)
    assert obj.shape == (12, 128)


def test_object() -> None:
    """Test generation of a single cell object in a frame."""
    obj = generat.gen_object(nrows=10, ncols=10, min_radius=2, max_radius=5, seed=1)
    assert obj[4, 4]
    assert not obj[1, 8]


def test_object_shape() -> None:
    """Test nrows and ncols."""
    obj = generat.gen_object(nrows=12)
    assert obj.shape == (12, 128)


class FrameData:
    """Group test frame data."""

    def __init__(self) -> None:
        self.bias = generat.gen_bias(ncols=6)
        self.flat = generat.gen_flat(ncols=4)
        self.objs = generat.gen_objs(generat.ImageObjsParams(ncols=64), seed=2)
        self.frame = generat.gen_frame(self.objs, seed=11)


@pytest.fixture(scope="class", name="frame_data")
def frame_data_setup() -> FrameData:
    """Generate frame data."""
    return FrameData()


class TestFrame:
    """Test frames."""

    def test_objs(self, frame_data: FrameData) -> None:
        """Test generation of a frame with objects."""
        assert frame_data.objs[104, 22] == 55.52683159241242

    def test_frame(self, frame_data: FrameData) -> None:
        """Test simulation of an acquired frame."""
        assert frame_data.frame[50, 31] == 2.0

    def test_frame_shape(self, frame_data: FrameData) -> None:
        """Test (nrows, ncols) shape."""
        assert frame_data.frame.shape == (128, 64)

    def test_frame_warn(self, frame_data: FrameData) -> None:
        """It warns if shape mismatch between objs and bias or flat."""
        with pytest.warns(UserWarning, match="Shape mismatch. Generate Bias..."):
            generat.gen_frame(frame_data.objs, frame_data.bias)
        with pytest.warns(UserWarning, match="Shape mismatch. Generate Flat..."):
            generat.gen_frame(frame_data.objs, flat=frame_data.flat)

    """
    Test simulation of an acquired frame.

    Attributes
    ----------
    bias : NDArray[np.float64]
        The bias array for the frame.
    flat : NDArray[np.float64]
        The flat field array for the frame.
    objs : NDArray[np.float64]
        The objects array for the frame.
    frame : NDArray[np.float64]
        The final simulated frame.
    """
