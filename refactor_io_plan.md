# NImA IO Refactoring Integration Plan

## Objective

Modernize the IO module of `nima` by replacing ad-hoc `tifffile` parsing with the standardized, plugin-based `bioio` library.

- Standardized `TCZYX` dimensional metadata.

- Lazy loading via Dask/Xarray.

- Pure Python (no JVM required for basic TIFF support).

- [x] `bioio` returns an `xarray.DataArray` with dimensions `TCZYX`.

- [x] Channels are a dimension (`C`), not keys in a dictionary.

- Leverage `BioImage`'s automatic Dask wrapping.

- Remove manual `TiffReader.aszarr()` boilerplate.

  Next Steps:

  - You can now proceed with migrating the internal logic of nima.py to use xarray/dask natively, eventually removing the .compute() call in the read_tiff shim.

bg can be redesigned?

Yes, dask-backed data types (and by extension xarray.DataArray wrapping them) would be more appropriate to maintain consistency with the library's move towards lazy evaluation and handling
of large datasets.

However, simply updating the type hint isn't enough. The functions in segmentation.py (like \_bg_arcsinh, \_bg_entropy) rely on scipy.ndimage and skimage.filters operations (e.g.,
percentile_filter, rank.entropy, threshold_local) which typically expect in-memory NumPy arrays. Passing a Dask array currently fails (as verified with a test script) because these
underlying libraries trigger immediate computation or don't support Dask arrays natively.

To support Dask arrays properly, these functions would need to be refactored to use dask.array.map_blocks or xarray.apply_ufunc, allowing them to process the image in chunks (with
appropriate overlaps/depth for filtering operations).

- See test_nima fixtures

NEXT:

- adopt dask_image instead of ndimage.median_filter
  Regarding DataArray.median(dim="T") vs dask-image:

  - Median Projection: DataArray.median(dim="T") is the correct, lazy approach for projections.
  - Median Filter (Spatial): dask-image.ndfilters.median_filter is superior for spatial filtering (denoising) of large-than-memory images, as ndimage.median_filter requires the whole array
    chunk in memory. This is a potential future optimization for nima.median.

- bi.mosaic_tile_dims

- devel docs -> cookie

- docs/conf.py -> cookie
