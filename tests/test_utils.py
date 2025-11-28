"""Tests for nima.utils."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from nima import utils
from nima.segmentation import BgResult, prob


def test_bg_estimates_gaussian_background() -> None:
    """Bg should fit deterministic data and return stable parameters."""
    frame = np.arange(16, dtype=np.float32).reshape(4, 4)

    estimated_mean, estimated_sd = utils.bg(frame)

    assert estimated_mean == pytest.approx(3.31, abs=0.5)
    assert estimated_sd == pytest.approx(3.90, abs=0.5)


def test_ave_masks_foreground(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ave should focus on bright objects after masking the background."""
    img = np.array(
        [
            [5.0, 20.0, 5.0],
            [20.0, 20.0, 5.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )
    bg_result = BgResult(bg=5.0, sd=2.0, iqr=(0.0, 0.0, 0.0), figures=None)

    monkeypatch.setattr(utils, "calculate_bg_iteratively", lambda _img: bg_result)

    result = utils.ave(img, bgmax=0.0, prob_value=0.001)

    mask = prob(img, bg_result.bg, bg_result.sd) < 0.001
    expected = float(img[mask].mean()) - bg_result.bg

    assert result == pytest.approx(expected)


def test_channel_mean_multichannel(monkeypatch: pytest.MonkeyPatch) -> None:
    """channel_mean should average each channel for 4-D data."""
    img = np.zeros((2, 3, 2, 2), dtype=np.float32)
    for t in range(img.shape[0]):
        for c in range(img.shape[1]):
            img[t, c] = np.full((2, 2), 10 * t + c, dtype=np.float32)

    def fake_ave(frame: np.ndarray, *_args: object, **_kwargs: object) -> float:
        return float(frame.mean())

    monkeypatch.setattr(utils, "ave", fake_ave)
    monkeypatch.setattr(utils, "_bgmax", lambda _img: 1.0)

    df = utils.channel_mean(img)

    assert list(df.columns) == ["0", "1", "2"]
    assert_allclose(
        df.iloc[0].to_numpy(dtype=np.float64),
        np.array([0.0, 1.0, 2.0], dtype=np.float64),
    )
    assert_allclose(
        df.iloc[1].to_numpy(dtype=np.float64),
        np.array([10.0, 11.0, 12.0], dtype=np.float64),
    )


def test_channel_mean_single_channel(monkeypatch: pytest.MonkeyPatch) -> None:
    """channel_mean should produce a YFP column for 3-D data."""
    img = np.zeros((2, 2, 2), dtype=np.float32)
    img[0] = 1.0
    img[1] = 3.0

    def fake_ave(frame: np.ndarray, *_args: object, **_kwargs: object) -> float:
        return float(frame.mean())

    monkeypatch.setattr(utils, "ave", fake_ave)

    df = utils.channel_mean(img)

    assert list(df.columns) == ["YFP"]
    assert_allclose(
        df["YFP"].to_numpy(dtype=np.float64), np.array([1.0, 3.0], dtype=np.float64)
    )


def test_ratio_df_with_yfp_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """ratio_df should normalize the YFP column when present."""
    monkeypatch.setattr(
        "nima.utils.tff.imread", lambda _path: np.ones((2, 2), dtype=np.float64)
    )

    def fake_channel_mean(_img: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame({"YFP": [10.0, 20.0, 30.0]}, dtype=np.float64)

    monkeypatch.setattr(utils, "channel_mean", fake_channel_mean)

    df = utils.ratio_df(["file1.tif", "file2.tif"])

    expected_norm = df["YFP"] / df["YFP"][:5].mean()
    assert_allclose(
        df["norm"].to_numpy(dtype=np.float64), expected_norm.to_numpy(dtype=np.float64)
    )


def test_ratio_df_adds_channel_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    """ratio_df should compute r_Cl and r_pH when channels 0, 1, 2 exist."""
    monkeypatch.setattr(
        "nima.utils.tff.imread", lambda _path: np.ones((2, 2), dtype=np.float64)
    )

    def fake_channel_mean(_img: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {0: [10.0, 20.0], 1: [5.0, 10.0], 2: [2.0, 4.0]}, dtype=np.float64
        )

    monkeypatch.setattr(utils, "channel_mean", fake_channel_mean)

    df = utils.ratio_df(["file1.tif"])

    assert_allclose(
        df["r_Cl"].to_numpy(dtype=np.float64),
        (df[2] / df[1]).to_numpy(dtype=np.float64),
    )
    assert_allclose(
        df["r_pH"].to_numpy(dtype=np.float64),
        (df[0] / df[2]).to_numpy(dtype=np.float64),
    )


def test_mask_all_channels_value_error() -> None:
    """mask_all_channels should raise when thresholds do not match channels."""
    image = np.ones((2, 2, 2), dtype=np.float32)
    with pytest.raises(
        ValueError, match="Number of thresholds must match the number of channels"
    ):
        utils.mask_all_channels(image, (0.5,))


def test_mask_all_channels_numpy_array() -> None:
    """mask_all_channels should combine thresholds for numpy arrays."""
    image = np.array(
        [
            [[1.0, 3.0], [4.0, 0.0]],
            [[2.0, 2.0], [3.0, 1.0]],
            [[3.0, 4.0], [5.0, 2.0]],
        ],
        dtype=np.float32,
    )
    thresholds = (0.5, 1.5, 2.5)

    result = utils.mask_all_channels(image, thresholds)
    expected = (
        (image[0] > thresholds[0])
        & (image[1] > thresholds[1])
        & (image[2] > thresholds[2])
    )

    assert result.dtype == bool
    assert_array_equal(result, expected)


def test_mask_all_channels_dask_array() -> None:
    """mask_all_channels should preserve dask arrays."""
    da = pytest.importorskip("dask.array")
    image_np = np.array(
        [
            [[0.0, 2.0], [3.0, 4.0]],
            [[1.0, 3.0], [2.0, 5.0]],
        ],
        dtype=np.float32,
    )
    image = da.from_array(image_np, chunks=image_np.shape)
    thresholds = (0.5, 1.5)

    result = utils.mask_all_channels(image, thresholds)

    assert hasattr(result, "compute")
    assert_array_equal(result.compute(), (image_np[0] > 0.5) & (image_np[1] > 1.5))
