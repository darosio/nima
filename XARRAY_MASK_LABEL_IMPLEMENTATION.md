# xarray.DataArray Support in `d_mask_label`

## Summary

The `d_mask_label` function in `src/nima/nima.py` has been updated to support `xarray.DataArray` inputs in addition to the legacy `DIm` dictionary format.

## Changes Made

### 1. New Internal Function: `_d_mask_label_xarray`

A new internal function that handles DataArray inputs:

```python
def _d_mask_label_xarray(
    d_im: xr.DataArray,
    min_size: int | None = 640,
    channels: tuple[str, ...] = ("C", "G", "R"),
    threshold_method: str = "yen",
    *,
    wiener: bool = False,
    watershed: bool = False,
    clear_border: bool = False,
    randomwalk: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
```

**Returns:** `(mask, labels)` as separate DataArrays

### 2. Updated `d_mask_label` Function

The main function now:

- Accepts `DIm | xr.DataArray` as input
- Dispatches to `_d_mask_label_xarray` for DataArray inputs
- Maintains backward compatibility with legacy DIm inputs
- Returns `tuple[xr.DataArray, xr.DataArray] | None`

**Return behavior:**

- **DataArray input:** Returns `(mask, labels)` tuple
- **DIm input:** Modifies in place, returns `None` (legacy behavior)

## API Design Decisions

### Why Return a Tuple Instead of Modifying DataArray?

Unlike dictionaries, xarray DataArrays cannot be modified in place by adding "keys" (which don't exist in DataArrays). The options were:

1. **Return tuple of (mask, labels)** ✅ **CHOSEN**

   - Clean functional API
   - Explicit return values
   - No side effects
   - Consistent with xarray conventions

1. ~~Return Dataset with mask and labels as variables~~

   - Would require converting the entire image data structure
   - More complex API change

1. ~~Expect a Dataset as input~~

   - Would break existing code that uses DataArray

### Dimensions

**Input:** `(T, C, Z, Y, X)`

- T: Time
- C: Channels (e.g., "C", "G", "R")
- Z: Z-stack (depth)
- Y: Height
- X: Width

**Output:** `(T, Z, Y, X)` for both mask and labels

- The channel dimension is reduced via geometric mean
- Mask is boolean (bool dtype)
- Labels are integer (int32 dtype)

## Implementation Details

### Algorithm Steps (vectorized)

All operations are applied plane-by-plane (Y, X) using `xr.apply_ufunc`:

1. **Geometric average** of specified channels:

   ```python
   ga = d_im.sel(C=list(channels)).prod(dim="C") ** (1 / len(channels))
   ```

1. **Optional Wiener filter** (3×3 kernel):

   ```python
   xr.apply_ufunc(signal.wiener, ga, ...)
   ```

1. **Thresholding** (yen or li method):

   ```python
   xr.apply_ufunc(lambda im: im > threshold_yen(im), ga_wiener, ...)
   ```

1. **Remove small objects**:

   ```python
   xr.apply_ufunc(morphology.remove_small_objects, mask, ...)
   ```

1. **Binary closing**:

   ```python
   xr.apply_ufunc(morphology.closing, mask, ...)
   ```

1. **Optional clear border**:

   ```python
   xr.apply_ufunc(segmentation.clear_border, mask, ...)
   ```

1. **Label connected components**:

   ```python
   xr.apply_ufunc(ndimage.label, mask, ...)
   ```

### Dask Support

All operations use `dask="parallelized"` in `apply_ufunc`, enabling:

- Lazy evaluation
- Parallel processing
- Memory-efficient computation on large datasets

### Not Yet Implemented

**Watershed segmentation** for DataArray inputs raises `NotImplementedError`:

```python
if watershed:
    msg = "Watershed not yet supported for DataArray input"
    raise NotImplementedError(msg)
```

The watershed algorithm requires complex region properties analysis and is more difficult to vectorize. It can be added in a future iteration.

## Usage Examples

### DataArray Input (New API)

```python
import xarray as xr
from nima.nima import d_mask_label

# Load image as DataArray (TCZYX)
d_im = xr.DataArray(...)

# Run segmentation
mask, labels = d_mask_label(
    d_im,
    min_size=640,
    channels=("C", "G", "R"),
    threshold_method="yen",
    wiener=False,
    clear_border=False
)

# mask: (T, Z, Y, X) boolean array
# labels: (T, Z, Y, X) int32 array
```

### DIm Input (Legacy API - Unchanged)

```python
from nima.nima import d_mask_label

# Legacy dictionary format
d_im = {
    "C": numpy_array,  # (T, Y, X)
    "G": numpy_array,
    "R": numpy_array,
}

# Modifies d_im in place, returns None
d_mask_label(d_im, min_size=640, channels=("C", "G", "R"))

# Access results via dictionary
mask = d_im["mask"]
labels = d_im["labels"]
```

## Testing

All existing tests pass without modification, confirming backward compatibility.

### Test Coverage

- ✅ Basic DataArray input/output
- ✅ Threshold methods (yen, li)
- ✅ Wiener filter option
- ✅ Clear border option
- ✅ Legacy DIm interface
- ✅ Error handling (invalid threshold method)
- ✅ NotImplementedError for watershed with DataArray

Run demonstration:

```bash
python test_xarray_mask_label.py
```

Run unit tests:

```bash
pytest tests/ -xvs
```

## Benefits

1. **Modern API:** Works naturally with xarray DataArrays
1. **Functional style:** No side effects for DataArray inputs
1. **Vectorized:** Efficient parallel processing via dask
1. **Backward compatible:** Legacy DIm interface unchanged
1. **Type safe:** Proper type hints for both input types

## Future Work

- Implement watershed segmentation for DataArray inputs
- Consider adding Dataset input/output option
- Add more comprehensive tests for edge cases
- Document performance characteristics with dask
