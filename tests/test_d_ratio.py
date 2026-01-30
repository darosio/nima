"""Tests for d_ratio function."""

import numpy as np
import xarray as xr

from nima import nima


def test_d_ratio_legacy() -> None:
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
    np.testing.assert_array_equal(d_im["r_cl"], expected)


def test_d_ratio_xarray() -> None:
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
