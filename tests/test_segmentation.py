"""Tests for segmentation module."""

import dask.array as da
import numpy as np
import pytest
import xarray as xr

from nima import segmentation
from nima.segmentation import BgParams


# Mock data
@pytest.fixture
def im() -> xr.DataArray:
    """Mock 100x100 image. Background: flat 10.0."""
    img = np.ones((100, 100), dtype=np.float64) * 10.0
    # Object: flat 100.0 in center 20:40, 20:40
    img[20:40, 20:40] = 100.0
    # Add slight noise to avoid constant value issues in some filters
    rng = np.random.default_rng(42)
    img += rng.normal(0, 0.1, img.shape)
    return xr.DataArray(img, dims=("y", "x"))


class TestSegmentationFunctions:
    """Test segmentation functions."""

    def test_bg_arcsinh(self, im: xr.DataArray) -> None:
        """Test arcsinh bg calculation."""
        params = BgParams(kind="arcsinh", radius=5, perc=10, arcsinh_perc=50)
        mask, _title, _lim = segmentation._bg_arcsinh(im, params)  # noqa: SLF001
        assert mask.shape == im.shape
        assert mask.dtype == bool
        # Corner should be background (True)
        assert mask[0, 0]
        # Center of object should be foreground (False)
        assert not mask[30, 30]

    def test_bg_entropy(self, im: xr.DataArray) -> None:
        """Test entropy bg calculation."""
        params = BgParams(kind="entropy", radius=5, perc=10)
        # Normalize to 0-1 range for entropy test to avoid skimage error
        # or use uint8 input
        im_norm = (im - im.min()) / (im.max() - im.min())
        mask, _title, _lim = segmentation._bg_entropy(im_norm, params)  # noqa: SLF001
        assert mask.shape == im.shape
        assert mask.dtype == bool
        # mask is now DataArray
        assert isinstance(mask, xr.DataArray)

    def test_bg_adaptive(self, im: xr.DataArray) -> None:
        """Test adaptive bg calculation."""
        params = BgParams(kind="adaptive", adaptive_radius=21)
        mask, _title, _ = segmentation._bg_adaptive(im, params)  # noqa: SLF001
        assert mask.shape == im.shape
        assert mask.dtype == bool
        # Check that the object is detected as foreground (False)
        assert not mask[30, 30]
        # Check that a significant portion is background (True)
        # In a noisy flat background, adaptive threshold with 0 offset splits it ~50/50
        # So we expect roughly 90% (bg area) * 50% = 45% background pixels
        assert mask.sum() > 0.4 * mask.size

    def test_bg_li_adaptive(self, im: xr.DataArray) -> None:
        """Test adaptive after li bg calculation."""
        params = BgParams(kind="li_adaptive", adaptive_radius=21)
        mask, _title, _ = segmentation._bg_li_adaptive(im, params)  # noqa: SLF001
        assert mask.shape == im.shape
        assert mask.dtype == bool
        assert not mask[30, 30]
        # Li + Adaptive should be better than just Adaptive for preserving background?
        # But adaptive step still cuts noise.
        assert mask.sum() > 0.4 * mask.size

    def test_bg_li_li(self, im: xr.DataArray) -> None:
        """Test li after li bg calculation."""
        params = BgParams(kind="li_li")
        mask, _title, _ = segmentation._bg_li_li(im, params)  # noqa: SLF001
        assert mask.shape == im.shape
        assert mask.dtype == bool
        assert not mask[30, 30]
        # Li-Li on flat background might result in empty or full mask depending on noise
        # Just check it returns a boolean mask

    def test_bg_inverse_yen(self, im: xr.DataArray) -> None:
        """Test inverse yen bg calculation."""
        # Inverse yen might be unstable on this synthetic data or produce different
        # results but let's check it runs.
        params = BgParams(kind="inverse_yen")
        # 1/im can be small.
        mask, _title, _ = segmentation._bg_inverse_yen(im, params)  # noqa: SLF001
        assert mask.shape == im.shape
        assert mask.dtype == bool

    def test_bg_dask_support(self, im: xr.DataArray) -> None:
        """Test that functions accept dask arrays and return dask arrays."""
        # Convert the fixture data to dask
        im_da = im.chunk({"y": 50, "x": 50})

        # 1. arcsinh
        params = BgParams(kind="arcsinh", radius=5, perc=10, arcsinh_perc=50)
        mask, _, _ = segmentation._bg_arcsinh(im_da, params)  # noqa: SLF001
        assert isinstance(mask.data, da.Array)
        assert mask.compute().shape == im.shape

        # 2. adaptive
        params = BgParams(kind="adaptive", adaptive_radius=21)
        mask, _, _ = segmentation._bg_adaptive(im_da, params)  # noqa: SLF001
        assert isinstance(mask.data, da.Array)
        assert mask.compute().shape == im.shape

        # 3. entropy
        params = BgParams(kind="entropy", radius=5, perc=10)
        # Normalize for entropy
        im_norm = (im_da - im_da.min()) / (im_da.max() - im_da.min())
        mask, _, _ = segmentation._bg_entropy(im_norm, params)  # noqa: SLF001
        assert isinstance(mask.data, da.Array)
        assert mask.compute().shape == im.shape

        # 4. li_adaptive
        params = BgParams(kind="li_adaptive", adaptive_radius=21, erosion_disk=1)
        mask, _, _ = segmentation._bg_li_adaptive(im_da, params)  # noqa: SLF001
        assert isinstance(mask.data, da.Array)

        # 5. li_li
        params = BgParams(kind="li_li", erosion_disk=1)
        mask, _, _ = segmentation._bg_li_li(im_da, params)  # noqa: SLF001
        assert isinstance(mask.data, da.Array)

        # 6. inverse_yen
        params = BgParams(kind="inverse_yen")
        # Avoid div by zero
        im_safe = im_da + 0.1
        mask, _, _ = segmentation._bg_inverse_yen(im_safe, params)  # noqa: SLF001
        assert isinstance(mask.data, da.Array)
