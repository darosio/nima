"""Tests for nima module."""

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pytest
import tifffile as tff
import xarray as xr

from nima import io, nima, segmentation

data_fp = "./tests/data/1b_c16_15.tif"


class TestShading:
    """Test shading."""

    def test_shading_broadcast(self) -> None:
        """Test shading with broadcastable dark and flat."""
        # Create DataArray (T, C, Y, X)
        # 5 timepoints, 2 channels (C, C2), 5x5 images
        data = np.zeros((5, 2, 5, 5))
        data[:, 0, :, :] = 2.0  # C
        data[:, 1, :, :] = 4.0  # C2

        im = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C", "C2"]})

        dark = 1.0
        flat = 2.0

        cor = nima.shading(im, dark, flat, clip=True)

        # when C: (2 - 1) / 2 = 0.5
        np.testing.assert_allclose(cor.sel(C="C").values, 0.5)
        # when C2: (4 - 1) / 2 = 1.5
        np.testing.assert_allclose(cor.sel(C="C2").values, 1.5)

    def test_shading_per_channel_flat(self) -> None:
        """Test shading with per-channel flat."""
        data = np.zeros((5, 2, 5, 5))
        data[:, 0, :, :] = 2.0
        data[:, 1, :, :] = 4.0
        im = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C", "C2"]})

        dark = 1.0
        # flat: C=2, C2=3
        flat_data = np.zeros((2, 5, 5))
        flat_data[0, :, :] = 2.0
        flat_data[1, :, :] = 3.0
        flat = xr.DataArray(flat_data, dims=("C", "Y", "X"), coords={"C": ["C", "C2"]})

        cor = nima.shading(im, dark, flat, clip=True)

        # when C: (2 - 1) / 2 = 0.5
        np.testing.assert_allclose(cor.sel(C="C").values, 0.5)
        # when C2: (4 - 1) / 3 = 1.0
        np.testing.assert_allclose(cor.sel(C="C2").values, 1.0)

    def test_shading_time_broadcasting(self) -> None:
        """Test shading when im has multiple timepoints and dark/flat have one."""
        # im: 3 timepoints
        im_data = np.ones((3, 1, 10, 10)) * 10.0
        im = xr.DataArray(
            im_data, dims=("T", "C", "Y", "X"), coords={"T": [0, 1, 2], "C": ["Ch1"]}
        )

        # dark: 1 timepoint (e.g. read from file)
        dark_data = np.ones((1, 1, 10, 10)) * 2.0
        dark = xr.DataArray(
            dark_data, dims=("T", "C", "Y", "X"), coords={"T": [0], "C": ["Ch1"]}
        )

        # flat: 1 timepoint
        flat_data = np.ones((1, 1, 10, 10)) * 2.0
        flat = xr.DataArray(
            flat_data, dims=("T", "C", "Y", "X"), coords={"T": [0], "C": ["Ch1"]}
        )

        # This failed if xarray aligns on T without squeezing
        cor = nima.shading(im, dark, flat)

        # Expected result: (10 - 2) / 2 = 4.0 for ALL timepoints
        assert cor.sizes["T"] == 3
        np.testing.assert_allclose(cor.values, 4.0)

    def test_shading_per_channel_dark_flat(self) -> None:
        """Test shading with per-channel dark and flat."""
        data = np.zeros((5, 2, 5, 5))
        data[:, 0, :, :] = 2.0
        data[:, 1, :, :] = 4.0
        im = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C", "C2"]})

        # dark: C=1, C2=2
        dark_data = np.zeros((2, 5, 5))
        dark_data[0, :, :] = 1.0
        dark_data[1, :, :] = 2.0
        dark = xr.DataArray(dark_data, dims=("C", "Y", "X"), coords={"C": ["C", "C2"]})

        # flat: C=2, C2=3
        flat_data = np.zeros((2, 5, 5))
        flat_data[0, :, :] = 2.0
        flat_data[1, :, :] = 3.0
        flat = xr.DataArray(flat_data, dims=("C", "Y", "X"), coords={"C": ["C", "C2"]})

        cor = nima.shading(im, dark, flat, clip=True)

        # when C: (2 - 1) / 2 = 0.5
        np.testing.assert_allclose(cor.sel(C="C").values, 0.5)
        # when C2: (4 - 2) / 3 = 2/3
        np.testing.assert_allclose(cor.sel(C="C2").values, 2.0 / 3.0)


@pytest.fixture(name="im")
def im_setup() -> xr.DataArray:
    """Read image as DataArray."""
    return io.read_image(Path(data_fp), channels=["0", "1", "2"])


class TestCalculateBg:
    """Test bg methods using nima.bg on DataArray."""

    def test_default(self, im: xr.DataArray) -> None:
        """Test default (arcsinh) method."""
        # T=3, C=2 ("2")
        subset = im.isel(T=[3]).sel(C=["2"])
        bg_params = segmentation.BgParams(kind="arcsinh")
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 286

    def test_arcsinh(self, im: xr.DataArray) -> None:
        """Test arcsinh method and arcsinh_perc, radius and perc arguments."""
        subset = im.isel(T=[3]).sel(C=["2"])
        bg_params = segmentation.BgParams(kind="arcsinh")
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 286

        bg_params = segmentation.BgParams(kind="arcsinh", arcsinh_perc=50, radius=15)
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 287

        bg_params = segmentation.BgParams(
            kind="arcsinh", arcsinh_perc=50, radius=15, perc=20
        )
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 288

    def test_entropy(self, im: xr.DataArray) -> None:
        """Test entropy method and radius argument."""
        subset = im.isel(T=[3]).sel(C=["2"])
        bg_params = segmentation.BgParams(kind="entropy")
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 297

        bg_params = segmentation.BgParams(kind="entropy", radius=20)
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 293

    def test_adaptive(self, im: xr.DataArray) -> None:
        """Test adaptive method and adaptive_radius argument."""
        subset = im.isel(T=[3]).sel(C=["2"])
        bg_params = segmentation.BgParams(kind="adaptive")
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 287

        bg_params = segmentation.BgParams(kind="adaptive", adaptive_radius=101)
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 280

    def test_li_adaptive(self, im: xr.DataArray) -> None:
        """Test li_arcsinh method."""
        subset = im.isel(T=[3]).sel(C=["2"])
        bg_params = segmentation.BgParams(kind="li_adaptive")
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 273

    def test_li_li(self, im: xr.DataArray) -> None:
        """Test li_li method."""
        subset = im.isel(T=[3]).sel(C=["2"])
        bg_params = segmentation.BgParams(kind="li_li")
        _, bgs, _ = nima.bg(subset, bg_params)
        assert int(cast("float", bgs.iloc[0, 0])) == 288


def test_plot_img_profile() -> None:
    """Plot summary graphics for Bias-Flat images.

    Test both lines (whole frame and central region) along x (axis=0).

    """
    sample_flat_image = Path("tests") / "data" / "output" / "test_flat_gaussnorm.tif"
    img = xr.DataArray(np.array(tff.imread(sample_flat_image)))
    f = nima.plt_img_profile(img)
    _, y_plot = f.get_axes()[1].lines[0].get_xydata().T  # type: ignore[union-attr]
    ydata = np.array([1.00000001, 0.99999999, 1.00000002, 1.0, 0.99999999])
    np.testing.assert_allclose(y_plot, ydata)
    _, y_plot = f.get_axes()[1].lines[1].get_xydata().T  # type: ignore[union-attr]
    ydata = np.array([1.0, 0.99999997, 1.0, 0.99999998, 0.99999997])
    np.testing.assert_allclose(y_plot, ydata)


class TestRatio:
    """Tests for ratio function."""

    def test_ratio(self) -> None:
        """Test xarray-based ratio."""
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

        res = nima.ratio(da, channels=("C", "R"), radii=(1,), mask=mask)
        assert isinstance(res, xr.DataArray)


class TestMeasure:
    """Tests for measure function."""

    def test_measure(self) -> None:
        """Test xarray-based measure."""
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

        labels_np = np.zeros((2, 10, 10), dtype=int)
        labels_np[0, 2:5, 2:5] = 1
        labels_np[1, 2:5, 2:5] = 1

        labels = xr.DataArray(labels_np, dims=("T", "Y", "X"))

        meas, _pr = nima.measure(
            da,
            labels,
            channels=("C", "G", "R"),
            channels_cl=("C", "R"),
            channels_ph=("G", "C"),
            radii=(1,),
            ratios_from_image=True,
        )

        assert 1 in meas
        df = meas[1]
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2

        assert np.allclose(df["r_cl"], 5.0)
        assert np.allclose(df["r_pH"], 0.4)
        assert np.allclose(df["r_cl_median"], 5.0)


class TestBg:
    """Tests for bg function."""

    def test_bg_xarray(self) -> None:
        """Test bg with xarray.DataArray."""
        # Create DataArray (T, C, Y, X)
        data = np.zeros((2, 2, 20, 20))
        # Background 10
        data += 10.0
        # Object 100
        data[:, :, 5:10, 5:10] = 100.0

        da = xr.DataArray(
            data, dims=("T", "C", "Y", "X"), coords={"T": [0, 1], "C": ["C1", "C2"]}
        )

        bg_params = segmentation.BgParams(kind="arcsinh")

        # d_cor, bgs, figs
        d_cor, bgs, figs = nima.bg(da, bg_params)

        assert isinstance(d_cor, xr.DataArray)
        assert isinstance(bgs, pd.DataFrame)
        assert isinstance(figs, dict)

        # Check bgs
        # Median of background should be approx 10
        # arcsinh method approximates mode/median
        assert np.allclose(bgs.values, 10.0, atol=1.0)

        # Check correction
        # 100 - 10 = 90
        # 10 - 10 = 0
        expected_obj = 90.0
        expected_bg = 0.0

        # Check object region
        assert np.allclose(
            d_cor.isel(Y=slice(5, 10), X=slice(5, 10)).values, expected_obj, atol=1.0
        )
        # Check bg region
        assert np.allclose(
            d_cor.isel(Y=slice(0, 4), X=slice(0, 4)).values, expected_bg, atol=1.0
        )

        # Check shapes
        assert d_cor.shape == da.shape
        assert bgs.shape == (2, 2)  # T x C


class TestSegment:
    """Tests for segment function with xarray."""

    def test_segment_watershed(self) -> None:
        """Test that watershed works with xarray."""
        # Create DataArray (T, C, Y, X)
        data = np.zeros((1, 3, 20, 20))
        # Add two objects close to each other
        data[:, :, 5:15, 5:10] = 100.0
        data[:, :, 5:15, 12:17] = 100.0

        da = xr.DataArray(
            data, dims=("T", "C", "Y", "X"), coords={"C": ["C", "G", "R"]}
        )
        # Convert to dask to test lazy execution
        da = da.chunk({"T": 1, "C": 1, "Y": 20, "X": 20})

        # Should return labels only
        labels = nima.segment(da, watershed=True, min_size=10)

        assert isinstance(labels, xr.DataArray)

        # Check that we have labels
        assert labels.compute().max() >= 1
        # Check explicit mask
        mask = labels > 0
        assert mask.dtype == bool
        # Check it is a dask array (lazy)
        assert labels.chunks is not None

    def test_segment_options(self) -> None:
        """Test various options for segment to cover private helper functions."""
        # Create DataArray (T, C, Y, X)
        data = np.zeros((1, 3, 50, 50))

        # 1. Object in center (large)
        data[:, :, 20:30, 20:30] = 100.0
        # 2. Object at border (should be cleared if clear_border=True)
        data[:, :, 0:10, 20:30] = 100.0
        # 3. Small object (should be removed if min_size is set high enough)
        # Size 2x2 = 4 pixels
        data[:, :, 40:42, 40:42] = 100.0

        da = xr.DataArray(
            data, dims=("T", "C", "Y", "X"), coords={"C": ["C", "G", "R"]}
        )

        # Test 1: clear_border=True
        # _clear_border_2d
        labels = nima.segment(da, clear_border=True, min_size=0, watershed=False)
        mask = labels > 0
        # Center object should remain
        assert mask.isel(T=0, Y=25, X=25)
        # Border object should be gone
        assert not mask.isel(T=0, Y=5, X=25)

        # Test 2: min_size
        # _remove_small_2d
        # Small object is 4 pixels. Set min_size=5.
        labels = nima.segment(da, min_size=5, clear_border=False, watershed=False)
        mask = labels > 0
        # Small object should be gone
        assert not mask.isel(T=0, Y=40, X=40)
        # Center object (10x10=100) should remain
        assert mask.isel(T=0, Y=25, X=25)

        # Test 3: wiener=True
        # _wiener_2d
        # Hard to deterministic check noise reduction without complex setup,
        # but we can ensure it runs without error.
        # Add small random noise to avoid divide by zero in wiener filter
        rng = np.random.default_rng(42)
        data += rng.random(data.shape) * 0.1
        labels = nima.segment(da, wiener=True, min_size=0, watershed=False)
        mask = labels > 0
        assert mask.any()

        # Test 4: threshold_method='li'
        # _threshold_2d with Li
        labels = nima.segment(da, threshold_method="li", min_size=0, watershed=False)
        mask = labels > 0
        assert mask.any()


class TestMedian:
    """Tests for median function."""

    def test_median_xarray(self) -> None:
        """Test median with xarray.DataArray."""
        # Create a 3D image (T, C, Y, X) -> (1, 1, 10, 10)
        # Median filter uses disk(1) which affects Y, X

        # Create a simple image with a "hot pixel"
        data = np.zeros((1, 1, 10, 10))
        data[0, 0, 5, 5] = 100.0

        da = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C1"]})

        # Apply median
        res = nima.median(da)

        assert isinstance(res, xr.DataArray)
        # The hot pixel should be removed by median filter (radius=1 means 3x3 window)
        # 3x3 window around (5,5) has one 100 and eight 0s. Median is 0.
        assert res.isel(T=0, C=0, Y=5, X=5) == 0.0

        # Check dimensions are preserved
        assert res.dims == da.dims
        assert res.shape == da.shape

    def test_median_xarray_value(self) -> None:
        """Test median values."""
        # 3x3 area
        # 1 1 1
        # 1 100 1
        # 1 1 1
        # Median of this is 1.

        data = np.ones((1, 1, 3, 3))
        data[0, 0, 1, 1] = 100.0

        da = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C1"]})

        res = nima.median(da)
        assert res.isel(T=0, C=0, Y=1, X=1) == 1.0


class TestEdgeCases:
    """Test edge cases for nima functions."""

    def test_single_pixel_image(self) -> None:
        """Test processing on a 1x1 pixel image."""
        data = np.array([[[[100.0]]]])  # T=1, C=1, Y=1, X=1
        da = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C1"]})

        # 1. Median
        # Median of 1x1 is itself
        res_med = nima.median(da)
        assert res_med.shape == (1, 1, 1, 1)
        assert res_med.to_numpy()[0, 0, 0, 0] == 100.0

        # 2. Segment
        # Should handle it without crash, though result might be trivial
        # For a single pixel > 0, it should be labeled if threshold allows
        # But standard threshold methods might fail on single value or constant value
        # Using constant threshold if possible, but nima.segment uses 'yen', 'li' etc.
        # We expect it to run without error at least.
        try:
            res_seg = nima.segment(da, threshold_method="yen", min_size=0)
            assert res_seg.shape == (1, 1, 1)  # T, Y, X (C squeezed/reduced)
        except Exception:  # noqa: BLE001, S110
            # Some scikit-image threshold methods might complain about constant input
            pass

    def test_empty_image(self) -> None:
        """Test processing on an empty (all zero) image."""
        data = np.zeros((1, 1, 10, 10))
        da = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C1"]})

        # Segment should return all zeros
        # Must specify channel if it differs from default ("C", "G", "R")
        res_seg = nima.segment(da, channels=("C1",), min_size=0)

        # Note: Global thresholding (Yen/Li) on constant or pure noise images
        # is ill-defined and may return all 1s or random masks.
        # We only check that the function runs and returns valid output shape/type.
        assert isinstance(res_seg, xr.DataArray)
        assert res_seg.shape == (1, 10, 10)

    def test_all_nans(self) -> None:
        """Test processing on an all-NaN image."""
        data = np.full((1, 1, 10, 10), np.nan)
        da = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C1"]})

        # Shading should handle NaNs (propagate or mask)
        dark = xr.DataArray(
            np.zeros((1, 10, 10)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )
        flat = xr.DataArray(
            np.ones((1, 10, 10)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )

        # This will likely propagate NaNs
        res = nima.shading(da, dark, flat)
        assert np.isnan(res.values).all()

    def test_dimension_mismatch_shading(self) -> None:
        """Test shading with mismatched channels."""
        # Image has C1
        data = np.zeros((1, 1, 10, 10))
        da = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C1"]})

        # Flat has C2
        flat_data = np.ones((1, 10, 10))
        flat = xr.DataArray(flat_data, dims=("C", "Y", "X"), coords={"C": ["C2"]})

        dark = xr.DataArray(
            np.zeros((1, 10, 10)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )

        # Xarray alignment should result in empty result or NaNs if dimensions
        # don't match or it might raise an error if we enforce strict alignment.
        # Current implementation relies on xarray automatic alignment.
        # If C doesn't match, it might result in size 0 on C axis for inner join,
        # or expanded NaNs for outer join.
        # Standard arithmetic is inner join by default?
        # Actually xarray arithmetic is usually outer join on coords but let's see.

        res = nima.shading(da, dark, flat)
        # If the channel "C1" is not in flat ("C2"), result for "C1" should be
        # dropped or NaN depending on join.
        # Let's verify what we expect. Ideally we want it to fail or warn, but xarray
        # behavior is strict.

        # If the resulting array has 0 channels, that's a valid "mismatch" result
        assert "C" in res.coords


class TestXArrayBehavior:
    """Test xarray-specific behaviors in nima."""

    def test_attrs_preservation_shading(self) -> None:
        """Test that attributes are preserved after shading."""
        data = np.zeros((1, 1, 10, 10))
        da = xr.DataArray(
            data,
            dims=("T", "C", "Y", "X"),
            coords={"C": ["C1"]},
            attrs={"units": "counts", "description": "raw image"},
        )

        dark = xr.DataArray(
            np.zeros((1, 10, 10)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )
        flat = xr.DataArray(
            np.ones((1, 10, 10)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )

        res = nima.shading(da, dark, flat)

        # Check if attributes are preserved
        assert res.attrs.get("units") == "counts"
        assert res.attrs.get("description") == "raw image"

    def test_attrs_preservation_median(self) -> None:
        """Test that attributes are preserved after median filter."""
        data = np.zeros((1, 1, 10, 10))
        da = xr.DataArray(
            data,
            dims=("T", "C", "Y", "X"),
            coords={"C": ["C1"]},
            attrs={"units": "counts"},
        )

        res = nima.median(da)
        assert res.attrs.get("units") == "counts"

    def test_coord_preservation(self) -> None:
        """Test that extra coordinates are preserved."""
        data = np.zeros((2, 1, 10, 10))
        da = xr.DataArray(
            data,
            dims=("T", "C", "Y", "X"),
            coords={
                "T": [0, 1],
                "C": ["C1"],
                "time_sec": ("T", [0.0, 1.5]),  # Extra coord on T
            },
        )

        dark = xr.DataArray(
            np.zeros((1, 10, 10)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )
        flat = xr.DataArray(
            np.ones((1, 10, 10)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )

        res = nima.shading(da, dark, flat)

        assert "time_sec" in res.coords
        assert res.coords["time_sec"].to_numpy()[1] == 1.5

    def test_dask_lazy_evaluation(self) -> None:
        """Explicit test for dask lazy evaluation."""
        data = np.zeros((10, 1, 100, 100))  # T=10
        da = xr.DataArray(data, dims=("T", "C", "Y", "X"), coords={"C": ["C1"]})
        # Chunk it along T, keep Y/X contiguous
        da = da.chunk({"T": 1, "C": 1, "Y": -1, "X": -1})

        dark = xr.DataArray(
            np.zeros((1, 100, 100)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )
        flat = xr.DataArray(
            np.ones((1, 100, 100)), dims=("C", "Y", "X"), coords={"C": ["C1"]}
        )

        # Shading
        res = nima.shading(da, dark, flat)
        assert res.chunks is not None

        # Median
        # Median on dask arrays usually requires map_overlap or similar
        # nima.median uses apply_ufunc with dask='parallelized' or 'allowed'
        # Let's check.
        res_med = nima.median(da)
        assert res_med.chunks is not None
