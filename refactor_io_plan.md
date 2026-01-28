# NImA IO Refactoring Integration Plan

## Objective

Modernize the IO module of `nima` by replacing ad-hoc `tifffile` parsing with the standardized, plugin-based `bioio` library.

## Recommended Architecture

### 1. Core Dependency

- **Library**: `bioio` + `bioio-tifffile`
- **Role**: Primary interface for image reading.
- **Benefits**:
  - Standardized `TCZYX` dimensional metadata.
  - Lazy loading via Dask/Xarray.
  - Pure Python (no JVM required for basic TIFF support).

### 2. Optional "Extra" Support

- **Library**: `bioio-bioformats` (depends on `jpype1`, `scyjava`)
- **Role**: Optional plugin for proprietary formats (LIF, CZI, ND2).
- **Strategy**: Expose as an optional install (e.g., `pip install nima[all_formats]`).

## Implementation Steps

### Step 1: Refactor IO Module

- Create `src/nima/io.py` (or update `nima.py`).
- Replace `read_tiff` and `read_tiffmd` with a generic `read_image` function.
- **New Pattern**:
  ```python
  from bioio import BioImage

  def read_image(fp: Path) -> xarray.DataArray:
      # Works for TIFF, and if plugins are installed, CZI/LIF too
      img = BioImage(fp)
      return img.get_image_data("TCZYX")
  ```

### Step 2: Standardize on Xarray

- **Legacy**: `DIm` (dictionary of images, e.g., `{'G': array, 'R': array}`).
- **Target**: `xarray.DataArray` or `xarray.Dataset`.
- **Action**:
  - `bioio` returns an `xarray.DataArray` with dimensions `TCZYX`.
  - Channels are a dimension (`C`), not keys in a dictionary.
  - Update `nima.py` functions that expect `DIm` to handle `DataArray` instead.
  - **Crucial**: Ratio imaging implies operations between channels (e.g., `G / R`). Xarray handles this via coordinate selection (e.g., `img.sel(C='G') / img.sel(C='R')`).

### Step 3: Lazy Loading

- Leverage `BioImage`'s automatic Dask wrapping.
- Remove manual `TiffReader.aszarr()` boilerplate.

### Step 4: Cleanup Dependencies

- **Remove**: `pims` (Done - unused), `impy-array` (redundant), `dask-image` (unless ndfilters needed).
- **Keep**:
  - `xmltodict` (used in `nima.py`).
  - `tifffile` (for low-level overrides).
  - `pyarrow` (Required for Pandas >= 3.0.0 and efficient data handling).
