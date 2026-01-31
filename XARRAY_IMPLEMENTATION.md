# xarray.Dataset Support in `nima`

## Summary

The `nima` library has been refactored to use `xarray.Dataset` as the primary data structure, replacing the legacy `DIm` (dictionary of images) and the intermediate `xarray.DataArray` support. This change enables robust metadata handling, simplified APIs, and efficient lazy evaluation with Dask.

## Changes Made

### 1. Core Functions Refactored

All core functions in `src/nima/nima.py` now accept `xarray.Dataset` as input and return `xarray.Dataset` (or compatible types). `xarray.DataArray` support has been removed from the public API to enforce consistency.

- **`d_median`**: Applies median filter to each variable in the Dataset.
- **`d_shading`**: Performs shading correction `(img - dark) / flat` using Dataset arithmetic.
- **`d_bg`**: Calculates background for each channel and subtracts it.
- **`d_mask_label`**: Computes mask and labels from multichannel data and returns a new Dataset with `mask` and `labels` variables.
- **`d_ratio`**: Computes ratio images (e.g., `r_cl = C / R`).
- **`d_meas_props`**: Measures properties (intensity, size, ratios) for labeled regions across time and channels.

### 2. Watershed Support

The `d_mask_label` function now fully supports watershed segmentation for `Dataset` inputs. The implementation uses `skimage.segmentation.watershed` (or `random_walker`) applied plane-by-plane via `xr.apply_ufunc`.

### 3. API Design

**Input:** `xr.Dataset`

- Dimensions: `(T, Z, Y, X)` (typically)
- Variables: Channel names (e.g., "C", "G", "R")

**Output:** `xr.Dataset`

- Preserves input structure.
- Adds new variables (e.g., `mask`, `labels`, `r_cl`) or returns modified copies.

This functional approach avoids side effects (unlike the legacy dictionary in-place modification).

## Performance Characteristics with Dask

The implementation leverages `xarray`'s integration with `dask` to provide lazy evaluation and parallel processing.

### Lazy Evaluation

Most operations (e.g., arithmetic, `d_median`, `d_bg`) use `dask` arrays under the hood. Computation is deferred until results are explicitly requested (e.g., `compute()`, plotting, or accessing values as numpy arrays).

### Parallelization

`xr.apply_ufunc` with `dask="parallelized"` is used for operations that apply to each 2D plane (e.g., filtering, thresholding, morphology). This allows Dask to:

1. **Chunking**: Process timepoints or Z-slices in parallel chunks.
1. **Memory Efficiency**: Load only necessary chunks into memory, enabling processing of datasets larger than RAM.

**Note:** Some operations, like `skimage.measure.regionprops` in `d_meas_props`, require eager evaluation (conversion to numpy arrays) for the specific frame being analyzed. However, the iteration over timepoints can still be efficient if the underlying data is chunked appropriately.

### Recommendations

- Use `chunks={'T': 1}` (or small blocks of T) when loading data to maximize parallelization over time.
- Avoid calling `.compute()` prematurely.

## Testing

The test suite `tests/test_nima.py` has been updated to use `xr.Dataset` fixtures. All tests pass, confirming the correctness of the refactoring.
The example script `examples/test_xarray_mask_label.py` demonstrates the new API usage.
