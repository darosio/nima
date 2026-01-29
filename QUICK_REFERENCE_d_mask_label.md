# Quick Reference: d_mask_label with xarray.DataArray

## Basic Usage

```python
import xarray as xr
from nima.nima import d_mask_label

# Load image as DataArray (TCZYX format)
d_im = xr.DataArray(...)

# Run segmentation - returns (mask, labels)
mask, labels = d_mask_label(
    d_im,
    min_size=640,                    # Minimum object size in pixels
    channels=("C", "G", "R"),        # Channels to use for segmentation
    threshold_method="yen",          # "yen" or "li"
    wiener=False,                    # Apply Wiener filter?
    clear_border=False,              # Remove border objects?
    watershed=False,                 # Not yet implemented for DataArray
)

# Output dimensions: (T, Z, Y, X)
# mask: boolean (True = foreground)
# labels: int32 (0 = background, 1+ = object IDs)
```

## Return Value Behavior

| Input Type     | Return Value                        | Side Effects                          |
| -------------- | ----------------------------------- | ------------------------------------- |
| `xr.DataArray` | `tuple[xr.DataArray, xr.DataArray]` | None                                  |
| `DIm` (dict)   | `None`                              | Adds 'mask' and 'labels' keys to dict |

## Options

### Threshold Methods

- `"yen"` (default): Yen's method - good for general use
- `"li"`: Li's method - may work better for low contrast

### Wiener Filter

```python
mask, labels = d_mask_label(d_im, wiener=True)
```

- Applies 3×3 Wiener filter for noise reduction
- Useful for noisy images

### Clear Border

```python
mask, labels = d_mask_label(d_im, clear_border=True)
```

- Removes objects touching image borders
- Useful for avoiding edge artifacts

### Min Size

```python
mask, labels = d_mask_label(d_im, min_size=100)
```

- Removes objects smaller than specified pixel count
- Set to `None` to keep all objects

## Common Use Cases

### Simple Segmentation

```python
# Basic segmentation with default settings
mask, labels = d_mask_label(d_im, channels=("C", "G", "R"))
```

### Noisy Images

```python
# Use Wiener filter for noisy data
mask, labels = d_mask_label(
    d_im,
    channels=("C", "G"),
    wiener=True,
    min_size=100
)
```

### Remove Small Debris and Border Cells

```python
mask, labels = d_mask_label(
    d_im,
    channels=("C", "G"),
    min_size=500,
    clear_border=True
)
```

## Working with Results

### Count Objects per Time Point

```python
mask, labels = d_mask_label(d_im, channels=("C", "G", "R"))

for t in range(labels.sizes["T"]):
    n_cells = len(np.unique(labels.isel(T=t))) - 1  # -1 for background
    print(f"Time {t}: {n_cells} cells")
```

### Extract Label at Specific Time/Z

```python
mask, labels = d_mask_label(d_im, channels=("C", "G"))

# Get labels at t=0, z=0
labels_t0z0 = labels.isel(T=0, Z=0).values  # numpy array
```

### Visualize Mask

```python
import matplotlib.pyplot as plt

mask, labels = d_mask_label(d_im, channels=("C", "G"))

plt.imshow(mask.isel(T=0, Z=0), cmap="gray")
plt.title("Binary Mask")
plt.show()
```

### Visualize Labels

```python
import matplotlib.pyplot as plt

mask, labels = d_mask_label(d_im, channels=("C", "G"))

plt.imshow(labels.isel(T=0, Z=0), cmap="tab20")
plt.title("Labeled Regions")
plt.colorbar()
plt.show()
```

## Error Handling

### Invalid Threshold Method

```python
try:
    mask, labels = d_mask_label(d_im, threshold_method="invalid")
except ValueError as e:
    print(f"Error: {e}")
    # Error: threshold_method must be one of ['yen', 'li']
```

### Watershed Not Implemented

```python
try:
    mask, labels = d_mask_label(d_im, watershed=True)
except NotImplementedError as e:
    print(f"Error: {e}")
    # Error: Watershed not yet supported for DataArray input
```

## Performance Tips

1. **Use dask arrays** for large datasets - operations are parallelized automatically
1. **Adjust min_size** based on your expected cell size
1. **Wiener filter** adds computational cost - only use if needed
1. Operations are plane-by-plane (vectorized over T, Z dimensions)

## Legacy DIm Interface

Still fully supported:

```python
# Dictionary format (old API)
d_im = {
    "C": numpy_array,  # shape: (T, Y, X)
    "G": numpy_array,
    "R": numpy_array,
}

# Modifies d_im in place, returns None
d_mask_label(d_im, channels=("C", "G", "R"))

# Access results
mask = d_im["mask"]
labels = d_im["labels"]
```

## See Also

- Full documentation: `XARRAY_MASK_LABEL_IMPLEMENTATION.md`
- Examples: `examples/test_xarray_mask_label.py`
- Implementation summary: `IMPLEMENTATION_SUMMARY.md`
