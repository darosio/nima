#!/usr/bin/env python
"""Demonstration and test of xarray support in d_mask_label.

This script demonstrates the new xarray.DataArray support in the d_mask_label function.
It shows:
1. Basic usage with DataArray input
2. Different threshold methods (yen, li)
3. Wiener filter option
4. Clear border option
5. Return value differences between DIm and DataArray inputs
"""

import numpy as np
import xarray as xr
from numpy.typing import NDArray

from nima.nima import d_mask_label


def create_test_data(
    n_time: int = 2,
    n_channels: int = 3,
    height: int = 100,
    width: int = 100,
    n_cells: int = 5,
) -> NDArray[np.float32]:
    """Create synthetic test data with bright circular regions (cells)."""
    rng = np.random.default_rng(42)

    data = np.zeros((n_time, n_channels, 1, height, width), dtype=np.float32)

    # Add "cells" - bright circular regions
    for t in range(n_time):
        for c in range(n_channels):
            for _i in range(n_cells):
                y_center = rng.integers(20, height - 20)
                x_center = rng.integers(20, width - 20)
                radius = 10

                y, x = np.ogrid[0:height, 0:width]
                mask_circle = (y - y_center) ** 2 + (x - x_center) ** 2 <= radius**2
                data[t, c, 0, mask_circle] = 200 + rng.random() * 50

    # Add background noise
    data += rng.random((n_time, n_channels, 1, height, width)) * 10

    return data


def example_1_basic_usage() -> None:
    """Demonstrate basic usage with DataArray input."""
    print("=" * 70)
    print("Example 1: Basic usage with DataArray input")
    print("=" * 70)

    # Create test data
    data = create_test_data(n_time=2, n_channels=3)

    # Create DataArray with proper coordinates
    d_im = xr.DataArray(
        data,
        dims=["T", "C", "Z", "Y", "X"],
        coords={
            "T": [0, 1],
            "C": ["C", "G", "R"],
            "Z": [0],
            "Y": np.arange(100),
            "X": np.arange(100),
        },
    )

    print("\nInput DataArray:")
    print(f"  Shape: {d_im.shape}")
    print(f"  Dims: {d_im.dims}")
    print(f"  Dtype: {d_im.dtype}")

    # Run segmentation
    mask, labels = d_mask_label(
        d_im,
        min_size=50,
        channels=("C", "G", "R"),
        threshold_method="yen",
    )

    print("\nOutput:")
    print(f"  Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"  Labels shape: {labels.shape}, dtype: {labels.dtype}")
    print(f"  Number of cells (t=0): {len(np.unique(labels.isel(T=0).values)) - 1}")
    print(f"  Number of cells (t=1): {len(np.unique(labels.isel(T=1).values)) - 1}")

    print("\n✓ Success!")


def example_2_threshold_methods() -> None:
    """Compare threshold methods (yen vs li)."""
    print("\n" + "=" * 70)
    print("Example 2: Comparing threshold methods (yen vs li)")
    print("=" * 70)

    data = create_test_data(n_time=1, n_channels=2, n_cells=10)
    d_im = xr.DataArray(
        data,
        dims=["T", "C", "Z", "Y", "X"],
        coords={
            "T": [0],
            "C": ["C", "G"],
            "Z": [0],
            "Y": np.arange(100),
            "X": np.arange(100),
        },
    )

    # Try yen threshold
    _mask_yen, labels_yen = d_mask_label(
        d_im, min_size=30, channels=("C", "G"), threshold_method="yen"
    )
    n_cells_yen = len(np.unique(labels_yen.values)) - 1

    # Try li threshold
    _mask_li, labels_li = d_mask_label(
        d_im, min_size=30, channels=("C", "G"), threshold_method="li"
    )
    n_cells_li = len(np.unique(labels_li.values)) - 1

    print("\nThreshold method comparison:")
    print(f"  'yen': {n_cells_yen} cells detected")
    print(f"  'li':  {n_cells_li} cells detected")

    print("\n✓ Success!")


def example_3_wiener_filter() -> None:
    """Demonstrate Wiener filter effect."""
    print("\n" + "=" * 70)
    print("Example 3: Effect of Wiener filter")
    print("=" * 70)

    rng = np.random.default_rng(42)
    data = create_test_data(n_time=1, n_channels=2, n_cells=8)
    # Add more noise to demonstrate filter effect
    data += rng.random((1, 2, 1, 100, 100)) * 30

    d_im = xr.DataArray(
        data,
        dims=["T", "C", "Z", "Y", "X"],
        coords={
            "T": [0],
            "C": ["C", "G"],
            "Z": [0],
            "Y": np.arange(100),
            "X": np.arange(100),
        },
    )

    # Without Wiener filter
    _mask_no_wiener, labels_no_wiener = d_mask_label(
        d_im, min_size=40, channels=("C", "G"), wiener=False
    )
    n_cells_no_wiener = len(np.unique(labels_no_wiener.values)) - 1

    # With Wiener filter
    _mask_wiener, labels_wiener = d_mask_label(
        d_im, min_size=40, channels=("C", "G"), wiener=True
    )
    n_cells_wiener = len(np.unique(labels_wiener.values)) - 1

    print("\nWiener filter effect:")
    print(f"  Without Wiener: {n_cells_no_wiener} cells")
    print(f"  With Wiener:    {n_cells_wiener} cells")

    print("\n✓ Success!")


def example_4_clear_border() -> None:
    """Demonstrate clear border option."""
    print("\n" + "=" * 70)
    print("Example 4: Clear border option")
    print("=" * 70)

    # Create data with cells at the border
    rng = np.random.default_rng(42)
    data = np.zeros((1, 2, 1, 100, 100), dtype=np.float32)

    # Add cells in center
    for c in range(2):
        y, x = np.ogrid[0:100, 0:100]
        mask_center = (y - 50) ** 2 + (x - 50) ** 2 <= 15**2
        data[0, c, 0, mask_center] = 250

        # Add cells at borders
        for border_pos in [(5, 50), (95, 50), (50, 5), (50, 95)]:
            mask_border = (y - border_pos[0]) ** 2 + (x - border_pos[1]) ** 2 <= 12**2
            data[0, c, 0, mask_border] = 250

    data += rng.random((1, 2, 1, 100, 100)) * 10

    d_im = xr.DataArray(
        data,
        dims=["T", "C", "Z", "Y", "X"],
        coords={
            "T": [0],
            "C": ["C", "G"],
            "Z": [0],
            "Y": np.arange(100),
            "X": np.arange(100),
        },
    )

    # Without clear_border
    _mask_no_clear, labels_no_clear = d_mask_label(
        d_im, min_size=50, channels=("C", "G"), clear_border=False
    )
    n_cells_no_clear = len(np.unique(labels_no_clear.values)) - 1

    # With clear_border
    _mask_clear, labels_clear = d_mask_label(
        d_im, min_size=50, channels=("C", "G"), clear_border=True
    )
    n_cells_clear = len(np.unique(labels_clear.values)) - 1

    print("\nClear border effect:")
    print(f"  Without clear_border: {n_cells_no_clear} cells (includes border cells)")
    print(f"  With clear_border:    {n_cells_clear} cells (border cells removed)")

    print("\n✓ Success!")


def example_5_legacy_interface() -> None:
    """Demonstrate legacy DIm interface compatibility."""
    print("\n" + "=" * 70)
    print("Example 5: Legacy DIm interface (backward compatibility)")
    print("=" * 70)

    # Create legacy dictionary format
    rng = np.random.default_rng(42)
    d_im_dict = {
        "C": rng.random((2, 100, 100)) * 100,
        "G": rng.random((2, 100, 100)) * 100,
        "R": rng.random((2, 100, 100)) * 100,
    }

    # Add some bright spots
    for ch in ["C", "G", "R"]:
        d_im_dict[ch][0, 40:60, 40:60] = 250
        d_im_dict[ch][1, 30:50, 30:50] = 250

    print("\nInput DIm (legacy format):")
    print(f"  Keys: {list(d_im_dict.keys())}")
    print(f"  Channel 'C' shape: {d_im_dict['C'].shape}")

    # Call d_mask_label with legacy interface
    result = d_mask_label(d_im_dict, min_size=100, channels=("C", "G", "R"))

    print("\nOutput:")
    print(f"  Return value: {result} (None for legacy interface)")
    print(f"  'mask' added to d_im: {'mask' in d_im_dict}")
    print(f"  'labels' added to d_im: {'labels' in d_im_dict}")
    print(f"  Mask shape: {d_im_dict['mask'].shape}")
    print(f"  Labels shape: {d_im_dict['labels'].shape}")

    print("\n✓ Success!")


def example_6_watershed_not_implemented() -> None:
    """Verify watershed raises NotImplementedError for DataArray."""
    print("\n" + "=" * 70)
    print("Example 6: Watershed with DataArray (not yet implemented)")
    print("=" * 70)

    data = create_test_data(n_time=1, n_channels=2, n_cells=5)
    d_im = xr.DataArray(
        data,
        dims=["T", "C", "Z", "Y", "X"],
        coords={
            "T": [0],
            "C": ["C", "G"],
            "Z": [0],
            "Y": np.arange(100),
            "X": np.arange(100),
        },
    )

    try:
        _mask, _labels = d_mask_label(d_im, channels=("C", "G"), watershed=True)
        print("\n✗ ERROR: Should have raised NotImplementedError!")
    except NotImplementedError as e:
        print("\n✓ Correctly raised NotImplementedError:")
        print(f"  {e}")


def main() -> None:
    """Run all examples."""
    print("\n" + "=" * 70)
    print("DEMONSTRATION: xarray.DataArray support in d_mask_label")
    print("=" * 70)

    example_1_basic_usage()
    example_2_threshold_methods()
    example_3_wiener_filter()
    example_4_clear_border()
    example_5_legacy_interface()
    example_6_watershed_not_implemented()

    print("\n" + "=" * 70)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nKey points:")
    print("  • DataArray input returns tuple (mask, labels)")
    print("  • DIm input modifies in place and returns None")
    print("  • All options (wiener, clear_border, threshold methods) work")
    print("  • Watershed not yet supported for DataArray")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
