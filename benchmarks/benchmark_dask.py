"""Benchmarks for Dask-based image processing operations."""

import time

import dask.array as da
import numpy as np
import xarray as xr

from nima import nima


def run_benchmark() -> None:
    """Run performance benchmarks for shading correction and median filtering."""
    print("Benchmarking Nima with Dask...")

    # 1. Create a large lazy array (e.g. t=100, c=3, y=2048, x=2048)
    # Total elements: 100 * 3 * 2048 * 2048 ~ 1.2 billion pixels
    # As float64: ~9.6 GB
    # We chunk along t and c, keeping y/x full (typical for microscopy)

    t_sz, c_sz, y_sz, x_sz = 10, 3, 2048, 2048

    print(f"Creating array: T={t_sz}, C={c_sz}, Y={y_sz}, X={x_sz}")

    dask_data = da.random.random((t_sz, c_sz, y_sz, x_sz), chunks=(1, 1, y_sz, x_sz))
    im = xr.DataArray(
        dask_data, dims=("T", "C", "Y", "X"), coords={"C": ["C1", "C2", "C3"]}
    )

    # 2. Shading Correction
    print("\n--- Shading Correction ---")
    rng = np.random.default_rng()
    dark = xr.DataArray(
        rng.random((c_sz, y_sz, x_sz)),
        dims=("C", "Y", "X"),
        coords={"C": ["C1", "C2", "C3"]},
    )
    flat = xr.DataArray(
        rng.random((c_sz, y_sz, x_sz)),
        dims=("C", "Y", "X"),
        coords={"C": ["C1", "C2", "C3"]},
    )

    start_time = time.time()
    # This just builds the graph
    res_shading = nima.shading(im, dark, flat)
    graph_time = time.time() - start_time
    print(f"Graph construction time: {graph_time:.4f}s")

    # Trigger computation for a subset to verify it works and measure throughput
    # We compute one timepoint
    print("Computing one timepoint...")
    start_time = time.time()
    _ = res_shading.isel(T=0).compute()
    compute_time = time.time() - start_time
    print(f"Computation time (T=1): {compute_time:.4f}s")

    # 3. Median Filter
    print("\n--- Median Filter ---")
    # Median involves applying a filter on Y,X planes.
    start_time = time.time()
    res_median = nima.median(im)
    graph_time = time.time() - start_time
    print(f"Graph construction time: {graph_time:.4f}s")

    print("Computing one timepoint (median)...")
    start_time = time.time()
    _ = res_median.isel(T=0, C=0).compute()
    compute_time = time.time() - start_time
    print(f"Computation time (T=1, C=1): {compute_time:.4f}s")


if __name__ == "__main__":
    run_benchmark()
