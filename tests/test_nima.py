"""Tests for nima module."""

from pathlib import Path

import numpy as np
import tifffile as tff
from numpy.testing import assert_array_equal

from nima import nima

data_fp = "./tests/data/1b_c16_15.tif"


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
        self.im = np.array(tff.imread(data_fp))

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
        assert nima.bg(self.im[3, 2], kind="entropy")[0] == 297
        assert nima.bg(self.im[3, 2], kind="entropy", radius=20)[0] == 293

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


def test_plot_img_profile() -> None:
    """Plot summary graphics for Bias-Flat images.

    Test both lines (whole frame and central region) along x (axis=0).

    """
    sample_flat_image = Path("tests") / "data" / "output" / "test_flat_gaussnorm.tif"
    img = np.array(tff.imread(sample_flat_image))
    f = nima.plt_img_profile(img)
    _, y_plot = f.get_axes()[1].lines[0].get_xydata().T  # type: ignore
    ydata = np.array([1.00000001, 0.99999999, 1.00000002, 1.0, 0.99999999])
    np.testing.assert_allclose(y_plot, ydata)
    _, y_plot = f.get_axes()[1].lines[1].get_xydata().T  # type: ignore
    ydata = np.array([1.0, 0.99999997, 1.0, 0.99999998, 0.99999997])
    np.testing.assert_allclose(y_plot, ydata)
