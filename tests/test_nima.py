"""Tests for nima module."""
import numpy as np
import pytest
import tifffile as tff  # type: ignore
from numpy.testing import assert_array_equal  # assert_allclose

from nima import nima


data_fp = "./tests/data/1b_c16_15.tif"


class TestZproject:
    """Tests zproject."""

    def setup_class(self) -> None:
        """Set up stack arrays."""
        self.im = np.ones((4, 2, 5))
        self.im[0] = np.ones((2, 5)) * 2
        self.im_int = np.ones((4, 2, 5)).astype(int)
        self.im_int[0] = np.ones((2, 5)) * 2
        self.im_int1 = np.ones((5, 2, 5)).astype(int)
        self.im_int1[0] = np.ones((2, 5)) * 2

    def test_median(self) -> None:
        """It calculates median==1 for an array of 1, 1, 1, 1, 1."""
        res = nima.zproject(self.im)
        print(res.dtype)  # FIXME: test do not print!
        assert_array_equal(res, np.ones((2, 5)))

    def test_median_integer(self) -> None:
        """It works with integers."""
        res = nima.zproject(self.im_int)
        res1 = nima.zproject(self.im_int1)
        assert res.dtype == np.dtype(int)
        assert res1.dtype == np.dtype(int)
        assert_array_equal(res, np.ones((2, 5)).astype(int))
        assert_array_equal(res1, np.ones((2, 5)).astype(int))

    def test_raise_exception_2d_input(self) -> None:
        """It raises exception ..."""
        with pytest.raises(
            ValueError,
            match=r"Input must be 3D-grayscale .*",
        ):
            nima.zproject(self.im[0])


class TestDShading:
    """Test d_shading."""

    def setup_class(self) -> None:
        """Set up stack arrays."""
        self.d_im = {"C": np.ones((5, 5, 5)) * 2, "C2": np.ones((5, 5, 5)) * 4}
        self.dark = np.ones((5, 5))
        self.flat = np.ones((5, 5)) * 2
        self.d_flat = {"C": self.flat, "C2": np.ones((5, 5)) * 3}
        self.d_dark = {"C": self.dark, "C2": np.ones((5, 5)) * 2}

    def test_single_dark_and_single_flat(self) -> None:
        """Using single dark and single flat images."""
        d_cor = nima.d_shading(self.d_im, self.dark, self.flat, clip=True)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)) * 1.5)

    def test_single_dark_and_d_flat(self) -> None:
        """Using single dark and a stack of flat images."""
        d_cor = nima.d_shading(self.d_im, self.dark, self.d_flat, clip=True)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)))

    def test_d_dark_and_d_flat(self) -> None:
        """Using stacks of dark and flat images."""
        d_cor = nima.d_shading(self.d_im, self.d_dark, self.d_flat, clip=True)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)) * 2 / 3)


class TestBg:
    """Test bg methods."""

    def setup_class(self) -> None:
        """Read test data."""
        self.im = tff.imread(data_fp)

    def test_default(self) -> None:
        """Test default (arcsinh) method."""
        assert nima.bg(self.im[3, 2])[0] == 286

    def test_arcsinh(self) -> None:
        """Test arcsinh method and arcsinh_perc, radius and perc arguments."""
        assert nima.bg(self.im[3, 2], kind="arcsinh")[0] == 286
        assert (
            nima.bg(self.im[3, 2], kind="arcsinh", arcsinh_perc=50, radius=15)[0] == 287
        )
        assert (
            nima.bg(self.im[3, 2], kind="arcsinh", arcsinh_perc=50, radius=15, perc=20)[
                0
            ]
            == 288
        )

    def test_entropy(self) -> None:
        """Test entropy method and radius argument."""
        assert nima.bg(self.im[3, 2], kind="entropy")[0] == 274
        assert nima.bg(self.im[3, 2], kind="entropy", radius=20)[0] == 280

    def test_adaptive(self) -> None:
        """Test adaptive method and adaptive_radius argument."""
        assert nima.bg(self.im[3, 2], kind="adaptive")[0] == 287
        assert nima.bg(self.im[3, 2], kind="adaptive", adaptive_radius=101)[0] == 280

    def test_li_adaptive(self) -> None:
        """Test li_arcsinh method."""
        assert nima.bg(self.im[3, 2], kind="li_adaptive")[0] == 273

    def test_li_li(self) -> None:
        """Test li_li method."""
        assert nima.bg(self.im[3, 2], kind="li_li")[0] == 288