"""
Tests for nimg module.
"""
import pytest
import numpy as np
from numpy.testing import assert_array_equal
# from numpy.testing import assert_array_equal, assert_allclose
import nimg
import nimg.scripts


class Test_zproject:

    def setup_class(self):
        self.im = np.ones((4, 2, 5))
        self.im[0] = np.ones((2, 5)) * 2
        self.im_int = np.ones((4, 2, 5)).astype(int)
        self.im_int[0] = np.ones((2, 5)) * 2
        self.im_int1 = np.ones((5, 2, 5)).astype(int)
        self.im_int1[0] = np.ones((2, 5)) * 2

    def test_median(self):
        res = nimg.zproject(self.im)
        print(res.dtype)
        assert_array_equal(res, np.ones((2, 5)))

    def test_median_integer(self):
        res = nimg.zproject(self.im_int)
        res1 = nimg.zproject(self.im_int1)
        assert res.dtype == np.dtype(int)
        assert res1.dtype == np.dtype(int)
        assert_array_equal(res, np.ones((2, 5)).astype(int))
        assert_array_equal(res1, np.ones((2, 5)).astype(int))

    def test_raise_exception_2D_input(self):
        with pytest.raises(AssertionError) as err:
            nimg.zproject(self.im[0])
        assert str(err.value) == "Input must be 3D-grayscale (pln, row, col)"


class Test_d_shading:

    def setup_class(self):
        self.d_im = {'C': np.ones((5, 5, 5)) * 2, 'C2': np.ones((5, 5, 5)) * 4}
        self.dark = np.ones((5, 5))
        self.flat = np.ones((5, 5)) * 2
        self.d_flat = {'C': self.flat, 'C2': np.ones((5, 5)) * 3}
        self.d_dark = {'C': self.dark, 'C2': np.ones((5, 5)) * 2}

    def test_single_dark_and_single_flat(self):
        d_cor = nimg.d_shading(self.d_im, self.dark, self.flat, clip=True)
        # assert_allclose(d_cor, np.ones((5,5,5)) / 2)
        assert_array_equal(d_cor['C'], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor['C2'], np.ones((5, 5, 5)) * 1.5)

    def test_single_dark_and_d_flat(self):
        d_cor = nimg.d_shading(self.d_im, self.dark, self.d_flat, clip=True)
        # assert_allclose(d_cor, np.ones((5,5,5)) / 2)
        assert_array_equal(d_cor['C'], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor['C2'], np.ones((5, 5, 5)))

    def test_d_dark_and_d_flat(self):
        d_cor = nimg.d_shading(self.d_im, self.d_dark, self.d_flat, clip=True)
        assert_array_equal(d_cor['C'], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor['C2'], np.ones((5, 5, 5)) * 2 / 3)
