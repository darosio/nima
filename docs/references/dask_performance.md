# Dask Performance and Best Practices

## Overview

`nima` leverages `xarray` and `dask` to support large bioimage datasets that do not fit in memory. By using lazy evaluation and chunked arrays, `nima` can process multi-terabyte datasets efficiently, provided that individual 2D planes (Y, X) fit in memory.

## Chunking Strategy

The recommended chunking strategy for `nima` is to parallelize across the **Time (T)**, **Channel (C)**, and **Z-stack (Z)** dimensions, while keeping the spatial dimensions **(Y, X)** contiguous.

### Recommended Chunks

```python
# Optimal for nima workflows
chunks = {
    "T": 1,
    "C": 1,
    "Z": 1,
    "Y": -1,  # Full size
    "X": -1   # Full size
}
```

### Why Y/X must be contiguous?

Core operations in `nima` such as `median` filtering and `segment` (thresholding, watershed) are inherently 2D spatial operations.

- **Median Filter**: Uses `scipy.ndimage.median_filter`. Splitting Y/X into chunks would require complex overlap handling (`map_overlap`) to avoid edge artifacts. Currently, `nima` processes full 2D planes.
- **Segmentation**: Thresholding methods (Yen, Li) and labeling (connected components) require global context of the 2D plane.

## Performance Benchmarks

Benchmarks were run on a standard workstation with the following parameters:

- **Array Size**: T=10, C=3, Y=2048, X=2048 (Float64)
- **Total Data**: ~1.2 Billion pixels (~9.6 GB raw)
- **Operation**: Compute single timepoint (T=1)

| Operation              | Time (per T=1) | Throughput (approx) | Notes                                        |
| ---------------------- | -------------- | ------------------- | -------------------------------------------- |
| **Shading Correction** | ~0.06s         | ~1.6 GB/s           | Fully parallel, limited by memory bandwidth. |
| **Median Filter**      | ~0.13s         | ~250 MB/s           | CPU-bound spatial convolution (3x3 disk).    |

*Note: Benchmarks vary by hardware.*

## Limitations

1. **Memory Usage**: The largest single 2D plane (C, Y, X) or (Y, X) must fit in memory. For a 2k x 2k float64 image, this is small (~32MB). For very large stitched images (e.g., 50k x 50k), this approach will hit OOM errors.
1. **Global Thresholding**: Calculating thresholds requires computing histograms of the full 2D plane.

## Best Practices

- **Loading Data**: When using `nima.io.read_image`, dask arrays are created automatically if supported by the backend.
- **Processing**: Pipelines are built lazily. Use `.compute()` only when you need the final result (e.g., plotting, saving).
- **Saving**: Use `to_netcdf` or `to_zarr` for efficient parallel writing of results.
