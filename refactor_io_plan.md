# NImA IO Refactoring Integration Plan

## Objective

Modernize the IO module of `nima` by replacing ad-hoc `tifffile` parsing with the standardized, plugin-based `bioio` library.

## Recommended Architecture

### 1. Core Dependency

- **Library**: `bioio` + `bioio-tifffile` + `bioio-ome-tiff`
- **Role**: Primary interface for image reading.
- **Benefits**:
  - Standardized `TCZYX` dimensional metadata.
  - Lazy loading via Dask/Xarray.
  - Pure Python (no JVM required for basic TIFF support).

### 2. Optional "Extra" Support

- **Library**: `bioio-bioformats` (depends on `jpype1`, `scyjava`) and other like `bioio-lif`
- **Role**: Optional plugin for proprietary formats (LIF, CZI, ND2).
- **Strategy**: Expose as an optional install (e.g., `pip install nima[all_formats]`).

## Implementation Steps

### Step 1: Refactor IO Module

- [x] Create `src/nima/io.py` (or update `nima.py`).
- [x] Replace `read_tiff` and `read_tiffmd` with a generic `read_image` function.

### Step 2: Standardize on Xarray

- **Legacy**: `DIm` (dictionary of images, e.g., `{'G': array, 'R': array}`).
- **Target**: `xarray.DataArray` or `xarray.Dataset`.
- **Action**:
  - [x] `bioio` returns an `xarray.DataArray` with dimensions `TCZYX`.
  - [x] Channels are a dimension (`C`), not keys in a dictionary.
  - [ ] Update `nima.py` functions that expect `DIm` to handle `DataArray` instead.
  - **Crucial**: Ratio imaging implies operations between channels (e.g., `G / R`). Xarray handles this via coordinate selection (e.g., `img.sel(C='G') / img.sel(C='R')`).

### Step 3: Lazy Loading

- Leverage `BioImage`'s automatic Dask wrapping.

- Remove manual `TiffReader.aszarr()` boilerplate.

  Next Steps:

  - You can now proceed with migrating the internal logic of nima.py to use xarray/dask natively, eventually removing the .compute() call in the read_tiff shim.

❯ would dask_image be better than ndimage.median_filter?

- Add more comprehensive tests for edge cases
- Document performance characteristics with dask

1. Add comprehensive unit tests specifically for xarray functionality
1. Performance benchmarking with large dask arrays
1. Consider adding Dataset input/output option

add lxml and check metadata

bg can be redesigned?

Yes, dask-backed data types (and by extension xarray.DataArray wrapping them) would be more appropriate to maintain consistency with the library's move towards lazy evaluation and handling
of large datasets.

However, simply updating the type hint isn't enough. The functions in segmentation.py (like \_bg_arcsinh, \_bg_entropy) rely on scipy.ndimage and skimage.filters operations (e.g.,
percentile_filter, rank.entropy, threshold_local) which typically expect in-memory NumPy arrays. Passing a Dask array currently fails (as verified with a test script) because these
underlying libraries trigger immediate computation or don't support Dask arrays natively.

To support Dask arrays properly, these functions would need to be refactored to use dask.array.map_blocks or xarray.apply_ufunc, allowing them to process the image in chunks (with
appropriate overlaps/depth for filtering operations).

NEXT:

- bi.mosaic_tile_dims
