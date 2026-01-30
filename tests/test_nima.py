"""Tests for nima module."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile as tff
import xarray as xr
from numpy.testing import assert_array_equal
from numpy.typing import NDArray

from nima import nima, segmentation
from nima.nima_types import DIm, ImFrame

data_fp = "./tests/data/1b_c16_15.tif"


@pytest.fixture(name="d_im")
def d_im_setup() -> dict[str, NDArray[np.float64]]:
    """Create a dict of images."""
    return {
        "C": (np.ones((5, 5, 5)) * 2).astype(np.float64),
        "C2": (np.ones((5, 5, 5)) * 4).astype(np.float64),
    }


@pytest.fixture(name="d_flat")
def d_flat_setup() -> dict[str, NDArray[np.float64]]:
    """Create a dict of flat images."""
    return {
        "C": (np.ones((5, 5)) * 2).astype(np.float64),
        "C2": (np.ones((5, 5)) * 3).astype(np.float64),
    }


@pytest.fixture(name="d_dark")
def d_dark_setup() -> dict[str, NDArray[np.float64]]:
    """Create a dict of bias images."""
    return {"C": np.ones((5, 5)), "C2": (np.ones((5, 5)) * 2).astype(np.float64)}


@pytest.fixture(name="dark")
def dark_setup() -> NDArray[np.float64]:
    """Create a dark image."""
    return np.ones((5, 5))


@pytest.fixture(name="flat")
def flat_setup() -> NDArray[np.float64]:
    """Create a flat image."""
    return (np.ones((5, 5)) * 2).astype(np.float64)


class TestDShading:
    """Test d_shading."""

    def test_single_dark_and_single_flat(
        self,
        d_im: DIm,
        dark: ImFrame,
        flat: ImFrame,
    ) -> None:
        """Test d_shading using single dark and single flat images."""
        d_cor = nima.d_shading(d_im, dark, flat, clip=True)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)) * 1.5)

    def test_single_dark_and_d_flat(
        self,
        d_im: DIm,
        dark: ImFrame,
        d_flat: DIm,
    ) -> None:
        """Test d_shading using single dark and a stack of flat images."""
        d_cor = nima.d_shading(d_im, dark, d_flat, clip=True)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)))

    def test_d_dark_and_d_flat(
        self,
        d_im: DIm,
        d_dark: DIm,
        d_flat: DIm,
    ) -> None:
        """Test d_shading using stacks of dark and flat images."""
        d_cor = nima.d_shading(d_im, d_dark, d_flat, clip=True)
        assert_array_equal(d_cor["C"], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor["C2"], np.ones((5, 5, 5)) * 2 / 3)


@pytest.fixture(name="im")
def im_setup() -> NDArray[np.int_] | NDArray[np.float64]:
    """Create a dict of images."""
    return np.array(tff.imread(data_fp))


class TestCalculateBg:
    """Test bg methods."""

    def test_default(self, im: NDArray[np.int_] | NDArray[np.float64]) -> None:
        """Test default (arcsinh) method."""
        assert segmentation.calculate_bg(im[3, 2]).iqr[1] == 286

    def test_arcsinh(self, im: NDArray[np.int_] | NDArray[np.float64]) -> None:
        """Test arcsinh method and arcsinh_perc, radius and perc arguments."""
        bg_params = segmentation.BgParams(kind="arcsinh")
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 286
        bg_params = segmentation.BgParams(kind="arcsinh", arcsinh_perc=50, radius=15)
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 287
        bg_params = segmentation.BgParams(
            kind="arcsinh", arcsinh_perc=50, radius=15, perc=20
        )
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 288

    def test_entropy(self, im: NDArray[np.int_] | NDArray[np.float64]) -> None:
        """Test entropy method and radius argument."""
        bg_params = segmentation.BgParams(kind="entropy")
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 297
        bg_params = segmentation.BgParams(kind="entropy", radius=20)
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 293

    def test_adaptive(self, im: NDArray[np.int_] | NDArray[np.float64]) -> None:
        """Test adaptive method and adaptive_radius argument."""
        bg_params = segmentation.BgParams(kind="adaptive")
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 287
        bg_params = segmentation.BgParams(kind="adaptive", adaptive_radius=101)
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 280

    def test_li_adaptive(self, im: NDArray[np.int_] | NDArray[np.float64]) -> None:
        """Test li_arcsinh method."""
        bg_params = segmentation.BgParams(kind="li_adaptive")
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 273

    def test_li_li(self, im: NDArray[np.int_] | NDArray[np.float64]) -> None:
        """Test li_li method."""
        bg_params = segmentation.BgParams(kind="li_li")
        assert segmentation.calculate_bg(im[3, 2], bg_params).iqr[1] == 288


def test_plot_img_profile() -> None:
    """Plot summary graphics for Bias-Flat images.

    Test both lines (whole frame and central region) along x (axis=0).

    """
    sample_flat_image = Path("tests") / "data" / "output" / "test_flat_gaussnorm.tif"
    img = np.array(tff.imread(sample_flat_image))
    f = nima.plt_img_profile(img)
    _, y_plot = f.get_axes()[1].lines[0].get_xydata().T  # type: ignore[union-attr]
    ydata = np.array([1.00000001, 0.99999999, 1.00000002, 1.0, 0.99999999])
    np.testing.assert_allclose(y_plot, ydata)
    _, y_plot = f.get_axes()[1].lines[1].get_xydata().T  # type: ignore[union-attr]
    ydata = np.array([1.0, 0.99999997, 1.0, 0.99999998, 0.99999997])
    np.testing.assert_allclose(y_plot, ydata)


class TestDRatio:
    """Tests for d_ratio function."""

    def test_d_ratio_legacy(self) -> None:
        """Test legacy dictionary-based d_ratio."""
        # Create dummy DIm
        d_im = {
            "C": np.array([[[10.0, 10.0], [10.0, 10.0]]]),
            "R": np.array([[[2.0, 2.0], [2.0, 2.0]]]),
            "mask": np.array([[[1, 1], [0, 0]]]),
        }
        nima.d_ratio(d_im, channels=("C", "R"), radii=(1,))

        assert "r_cl" in d_im
        expected = np.array([[[5.0, 5.0], [0.0, 0.0]]])
        # The mask should zero out the second row
        assert_array_equal(d_im["r_cl"], expected)

    def test_d_ratio_xarray(self) -> None:
        """Test xarray-based d_ratio."""
        # data shapes: (1, 2, 2) -> (T, Y, X)
        data = np.array([[[10.0, 10.0], [10.0, 10.0]]])
        data2 = np.array([[[2.0, 2.0], [2.0, 2.0]]])

        # stack to get (T, C, Y, X) -> (1, 2, 2, 2)
        da = xr.DataArray(
            np.stack([data, data2], axis=1),
            dims=("T", "C", "Y", "X"),
            coords={"C": ["C", "R"]},
        )

        # We might need to pass mask if we want parity
        mask = xr.DataArray(np.array([[[1, 1], [0, 0]]]), dims=("T", "Y", "X"))

        res = nima.d_ratio(da, channels=("C", "R"), radii=(1,), mask=mask)
        assert res is not None
        assert isinstance(res, xr.DataArray)


class TestDMeasProps:
    """Tests for d_meas_props function."""

    def test_d_meas_props_legacy(self) -> None:
        """Test legacy dictionary-based d_meas_props."""
        # Create dummy DIm
        # 2 timepoints, 2 channels, 10x10 images
        # Time 0: Label 1 present
        # Time 1: Label 1 present
        labels = np.zeros((2, 10, 10), dtype=int)
        labels[0, 2:5, 2:5] = 1
        labels[1, 2:5, 2:5] = 1

        # C channel: constant 10
        c_ch = np.ones((2, 10, 10)) * 10.0
        # R channel: constant 2
        r_ch = np.ones((2, 10, 10)) * 2.0
        # G channel: constant 4
        g_ch = np.ones((2, 10, 10)) * 4.0

        d_im = {
            "C": c_ch,
            "R": r_ch,
            "G": g_ch,
            "labels": labels,
            "mask": labels > 0,  # d_ratio needs mask
        }

        meas, _pr = nima.d_meas_props(
            d_im,
            channels=("C", "G", "R"),
            channels_cl=("C", "R"),
            channels_ph=("G", "C"),
            radii=(1,),
            ratios_from_image=True,
        )

        assert 1 in meas
        df = meas[1]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # 2 timepoints

        # Check values
        # C=10, R=2. r_cl = 10/2 = 5
        # G=4, C=10. r_ph = 4/10 = 0.4

        # Check calculated ratios (from mean intensities)
        assert np.allclose(df["r_cl"], 5.0)
        assert np.allclose(df["r_pH"], 0.4)

        # Check ratios from image (median)
        # The ratio image should be 5.0 everywhere (ideal case)
        assert "r_cl_median" in df.columns
        assert np.allclose(df["r_cl_median"], 5.0)

    def test_d_meas_props_xarray(self) -> None:
        """Test xarray-based d_meas_props."""
        # Create DataArray (T, C, Y, X)
        # 2 timepoints, 3 channels (C, G, R), 10x10
        data = np.zeros((2, 3, 10, 10))
        # C channel (idx 0) = 10
        data[:, 0, :, :] = 10.0
        # G channel (idx 1) = 4
        data[:, 1, :, :] = 4.0
        # R channel (idx 2) = 2
        data[:, 2, :, :] = 2.0

        da = xr.DataArray(
            data, dims=("T", "C", "Y", "X"), coords={"C": ["C", "G", "R"]}
        )

        # Labels (T, Y, X) - note: legacy labels are usually just (T, Y, X) for 2D
        # timeseries. d_mask_label returns (T, Z, Y, X) but let's handle 3D or 4D
        # labels. If the input d_im is (T, C, Y, X), labels likely (T, Y, X).

        labels_np = np.zeros((2, 10, 10), dtype=int)
        labels_np[0, 2:5, 2:5] = 1
        labels_np[1, 2:5, 2:5] = 1

        labels = xr.DataArray(labels_np, dims=("T", "Y", "X"))

        # Attempt to call d_meas_props
        # Note: we need to pass labels explicitly or have them in d_im if it was a
        # dataset (but we assume DA input)

        try:
            meas, _pr = nima.d_meas_props(
                da,
                channels=("C", "G", "R"),
                channels_cl=("C", "R"),
                channels_ph=("G", "C"),
                radii=(1,),
                ratios_from_image=True,
                labels=labels,
            )

            assert 1 in meas
            df = meas[1]
            assert isinstance(df, pd.DataFrame)
            assert len(df) == 2

            assert np.allclose(df["r_cl"], 5.0)
            assert np.allclose(df["r_pH"], 0.4)
            assert np.allclose(df["r_cl_median"], 5.0)

        except TypeError as e:
            pytest.fail(f"d_meas_props failed with xarray: {e}")
        except NotImplementedError:
            pytest.fail("d_meas_props xarray support not implemented")
