"""Tests for generat module."""

import numpy as np
import pytest
from numpy import typing as npt

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
    np.random.seed(111)
    obj = generat.gen_object(nrows=10, ncols=10, min_radius=2, max_radius=5)
    assert obj[4, 4]
    assert not obj[8, 8]


def test_object_shape() -> None:
    """Test nrows and ncols."""
    obj = generat.gen_object(nrows=12)
    assert obj.shape == (12, 128)


class TestFrame:
    """Test simulation of an acquired frame."""

    bias: npt.NDArray[np.float_]
    flat: npt.NDArray[np.float_]
    objs: npt.NDArray[np.float_]
    frame: npt.NDArray[np.float_]

    def setup_class(self) -> None:
        """Read test data."""
        self.bias = generat.gen_bias(ncols=6)
        self.flat = generat.gen_flat(ncols=4)
        np.random.seed(121)
        self.objs = generat.gen_objs(ncols=64)
        self.frame = generat.gen_frame(self.objs)

    def test_objs(self) -> None:
        """Test generation of a frame with objects."""
        np.random.seed(111)
        assert generat.gen_objs()[23, 16] == 14.245461722859172

    def test_frame(self) -> None:
        """Test simulation of an acquired frame."""
        np.random.seed(121)
        assert generat.gen_frame(self.objs)[50, 31] == 24.46459269580741

    def test_frame_shape(self) -> None:
        """Test (nrows, ncols) shape."""
        assert self.frame.shape == (128, 64)

    def test_frame_warn(self) -> None:
        """It warns if shape mismatch between objs and bias or flat."""
        with pytest.warns(UserWarning, match="Shape mismatch. Generate Bias..."):
            generat.gen_frame(self.objs, self.bias)
        with pytest.warns(UserWarning, match="Shape mismatch. Generate Flat..."):
            generat.gen_frame(self.objs, flat=self.flat)
