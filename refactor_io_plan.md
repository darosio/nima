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

NEXT:

- bi.mosaic_tile_dims

- devel docs -> cookie

- docs/conf.py -> cookie
