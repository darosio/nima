"""Tests for nimg module."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal  # assert_allclose

import nimg.nimg as ni


class Test_zproject:
    """Tests zproject."""

    def setup_class(self):
        """Set up stack arrays."""
        self.im = np.ones((4, 2, 5))
        self.im[0] = np.ones((2, 5)) * 2
        self.im_int = np.ones((4, 2, 5)).astype(int)
        self.im_int[0] = np.ones((2, 5)) * 2
        self.im_int1 = np.ones((5, 2, 5)).astype(int)
        self.im_int1[0] = np.ones((2, 5)) * 2

    def test_median(self):
        """It calculates median==1 for an array of 1, 1, 1, 1, 1."""
        res = ni.zproject(self.im)
        print(res.dtype)  # FIXME: test do not print!
        assert_array_equal(res, np.ones((2, 5)))

    def test_median_integer(self):
        """It works with integers."""
        res = ni.zproject(self.im_int)
        res1 = ni.zproject(self.im_int1)
        assert res.dtype == np.dtype(int)
        assert res1.dtype == np.dtype(int)
        assert_array_equal(res, np.ones((2, 5)).astype(int))
        assert_array_equal(res1, np.ones((2, 5)).astype(int))

    def test_raise_exception_2D_input(self):
        """It raises exception ..."""
        with pytest.raises(AssertionError) as err:
            ni.zproject(self.im[0])
        assert str(err.value) == "Input must be 3D-grayscale (pln, row, col)"


class Test_d_shading:
    """Test d_shading."""

    def setup_class(self):
        """Set up stack arrays."""
        self.d_im = {"C": np.ones((5, 5, 5)) * 2, "C2": np.ones((5, 5, 5)) * 4}
        self.dark = np.ones((5, 5))
        self.flat = np.ones((5, 5)) * 2
        self.d_flat = {"C": self.flat, "C2": np.ones((5, 5)) * 3}
        self.d_dark = {"C": self.dark, "C2": np.ones((5, 5)) * 2}

    def test_single_dark_and_single_flat(self):
        """Using single dark and single flat images."""
        d_cor = ni.d_shading(self.d_im, self.dark, self.flat, clip=True)
        # assert_allclose(d_cor, np.ones((5,5,5)) / 2)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)) * 1.5)

    def test_single_dark_and_d_flat(self):
        """Using single dark and a stack of flat images."""
        d_cor = ni.d_shading(self.d_im, self.dark, self.d_flat, clip=True)
        # assert_allclose(d_cor, np.ones((5,5,5)) / 2)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)))

    def test_d_dark_and_d_flat(self):
        """Using stacks of dark and flat images."""
        d_cor = ni.d_shading(self.d_im, self.d_dark, self.d_flat, clip=True)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)) * 2 / 3)
