"""Generate mock images."""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import NDArray

from nima import segmentation, utils


def gen_bias(nrows: int = 128, ncols: int = 128) -> NDArray[np.float64]:
    """Generate a bias frame."""
    xvec = np.arange(ncols)
    yvec = 2 - (xvec**2 / 2.6 * (np.sin(xvec / 20) ** 2 + 0.1)) / 4000
    return np.tile(yvec * 2, (nrows, 1))


def gen_flat(nrows: int = 128, ncols: int = 128) -> NDArray[np.float64]:
    """Generate a flat frame."""
    y_idx, x_idx = np.mgrid[:nrows, :ncols]
    img = np.empty((nrows, ncols), dtype=float)
    img[:, :] = (
        -0.0001 * x_idx**2
        - 0.0001 * y_idx**2
        + 0.016 * y_idx
        + 0.014 * x_idx
        + 0.5
        + 0.000015 * x_idx * y_idx
    )
    img /= img.mean()
    return img


def gen_object(
    nrows: int = 128,
    ncols: int = 128,
    min_radius: int = 6,
    max_radius: int = 12,
    seed: int | None = None,
) -> NDArray[np.bool_]:
    """Generate a single ellipsoid object with random shape and position."""
    # Inspired by http://scipy-lectures.org/packages/scikit-image/index.html.
    x_idx, y_idx = np.indices((nrows, ncols))
    rng = np.random.default_rng(seed)
    x_obj, y_obj = rng.integers(0, nrows), rng.integers(0, ncols)
    radius = rng.integers(min_radius, max_radius)
    ellipse_factor = rng.random() * 3.5 - 1.75
    mask = (x_idx - x_obj) ** 2 + (y_idx - y_obj) ** 2 + ellipse_factor * (
        x_idx - x_obj
    ) * (y_idx - y_obj) < radius**2
    return np.asarray(mask, dtype=np.bool_)


# MAYBE: Convert to comments like #: Attribute desc
@dataclass
class ImageObjsParams:
    """
    Parameters for an image frame.

    Parameters
    ----------
    max_num_objects : int, optional
        Maximum number of objects to generate (default: 8).
    nrows : int, optional
        Number of rows in the image frame (default: 128).
    ncols : int, optional
        Number of columns in the image frame (default: 128).
    min_radius : int, optional
        Minimum radius of an object (default: 6).
    max_radius : int, optional
        Maximum radius of an object (default: 12).
    max_fluor : float, optional
        Maximum fluorescence intensity of an object (default: 20.0).
    """

    max_num_objects: int = 8
    nrows: int = 128
    ncols: int = 128
    min_radius: int = 6
    max_radius: int = 12
    max_fluor: float = 20.0


def gen_objs(
    params: ImageObjsParams | None = None, seed: int | None = None
) -> NDArray[np.float64]:
    """Generate a frame with ellipsoid objects; random n, shape, position and I."""
    params = ImageObjsParams() if params is None else params
    rng = np.random.default_rng(seed)
    min_num_objects = 2
    num_objs = (
        rng.integers(min_num_objects, params.max_num_objects)  # nosec "no-secure-random"
        if params.max_num_objects > min_num_objects
        else params.max_num_objects
    )
    # MAYBE: convolve the obj to simulate lower peri-cellular profile
    objs = [
        params.max_fluor
        * rng.random()
        * gen_object(
            params.nrows, params.ncols, params.min_radius, params.max_radius, seed
        )
        for _ in range(num_objs)
    ]
    return np.sum(objs, axis=0)  # type: ignore[no-any-return]


def gen_frame(  # noqa: PLR0913
    objs: NDArray[np.float64],
    bias: NDArray[np.float64] | None = None,
    flat: NDArray[np.float64] | None = None,
    sky: float = 2,
    noise_sd: float = 1,
    seed: int | None = None,
) -> NDArray[np.float64]:
    """Simulate an acquired frame [bias + noise + flat * (sky + obj)]."""
    (nrows, ncols) = objs.shape
    if bias is None:
        bias = np.zeros_like(objs)
    elif bias.shape != (nrows, ncols):
        warnings.warn("Shape mismatch. Generate Bias...", UserWarning, stacklevel=2)
        bias = gen_bias(nrows, ncols)
    if flat is None:
        flat = np.ones_like(objs)
    elif flat.shape != (nrows, ncols):
        warnings.warn("Shape mismatch. Generate Flat...", UserWarning, stacklevel=2)
        flat = gen_flat(nrows, ncols)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_sd, size=(nrows, ncols))
    img = bias + flat * (sky + objs) + noise
    return img.clip(0).astype("uint16")


def safe_call(
    func: Callable[..., Any],
    *args: Any,  # noqa: ANN401
    **kwargs: Any,  # noqa: ANN401
) -> Any:  # noqa: ANN401
    """Wrap a function call to handle exceptions.

    Parameters
    ----------
    func : Callable[..., Any]
        Function to call.
    *args : Any
        Positional arguments for func.
    **kwargs : Any
        Keyword arguments for func.

    Returns
    -------
    Any
        Result of the function call, or np.nan if an exception occurs.
    """
    try:
        return func(*args, **kwargs)
    except Exception:  # noqa: BLE001
        return np.nan


def run_simulation(  # noqa: PLR0913
    num_repeats: int = 1,
    noise_sd: float = 4,
    objs_pars: ImageObjsParams | None = None,
    bias: NDArray[np.float64] | None = None,
    flat: NDArray[np.float64] | None = None,
    sky: float = 10,
) -> pd.DataFrame:
    """Run background estimation simulation.

    Parameters
    ----------
    num_repeats : int
        Number of simulation iterations.
    noise_sd : float
        Standard deviation of noise to add to generated images.
    objs_pars : ImageObjsParams | None, optional
        Parameters for object generation.
    bias : NDArray[np.float64] | None, optional
        Bias frame.
    flat : NDArray[np.float64] | None, optional
        Flat frame.
    sky : float, optional
        Sky value (default: 10).

    Returns
    -------
    pd.DataFrame
        DataFrame containing estimated backgrounds and failure counts.
    """
    # Define configurations
    # Each config: (name, function, kwargs, extractor)
    configs: list[
        tuple[str, Callable[..., Any], dict[str, Any], Callable[[Any], float]]
    ] = [
        (
            "arcsinh",
            segmentation.calculate_bg,
            {"bg_params": segmentation.BgParams(kind="arcsinh")},
            lambda res: res.bg,
        ),
        (
            "entropy",
            segmentation.calculate_bg,
            {"bg_params": segmentation.BgParams(kind="entropy")},
            lambda res: res.bg,
        ),
        (
            "adaptive",
            segmentation.calculate_bg,
            {"bg_params": segmentation.BgParams(kind="adaptive")},
            lambda res: res.bg,
        ),
        (
            "li_adaptive",
            segmentation.calculate_bg,
            {"bg_params": segmentation.BgParams(kind="li_adaptive")},
            lambda res: res.bg,
        ),
        (
            "li_li",
            segmentation.calculate_bg,
            {"bg_params": segmentation.BgParams(kind="li_li")},
            lambda res: res.bg,
        ),
        (
            "inverse_yen",
            segmentation.calculate_bg,
            {"bg_params": segmentation.BgParams(kind="inverse_yen")},
            lambda res: res.bg,
        ),
        ("bg", segmentation.calculate_bg_iteratively, {}, lambda res: res.bg),
        ("bg2", utils.bg, {"bgmax": 800}, lambda res: res[0]),
    ]

    results: dict[str, list[float]] = {name: [] for name, _, _, _ in configs}
    failures = {name: 0 for name, _, _, _ in configs}

    if objs_pars is None:
        objs_pars = ImageObjsParams(max_num_objects=10, max_fluor=100, nrows=128)

    for _ in range(num_repeats):
        # Generate data
        objs = gen_objs(objs_pars)
        frame_np = gen_frame(objs, bias, flat, sky=sky, noise_sd=noise_sd)

        # Create DataArray (as expected by segmentation functions)
        # Note: gen_frame returns 2D numpy array
        frame_da = xr.DataArray(
            frame_np[np.newaxis, np.newaxis, ...],
            coords={
                "T": [10],
                "C": ["G"],
                "Y": range(objs_pars.nrows),
                "X": range(objs_pars.ncols),
            },
            dims=["T", "C", "Y", "X"],
        ).squeeze()

        for name, func, kwargs, extractor in configs:
            # Prepare arguments based on function signature
            args: tuple[Any, ...]
            if func in (utils.bg, segmentation.calculate_bg_iteratively):
                args = (frame_da.to_numpy(),)
            else:
                args = (frame_da,)

            res = safe_call(func, *args, **kwargs)

            val = np.nan
            is_failure = False

            # Check if res is nan failure (assuming valid result is not nan float)
            if isinstance(res, float) and np.isnan(res):
                is_failure = True
            else:
                try:
                    val = extractor(res)
                    # Check if extracted value is nan
                    if np.isnan(val):
                        is_failure = True
                except Exception:  # noqa: BLE001
                    is_failure = True

            if is_failure:
                failures[name] += 1
                val = np.nan

            results[name].append(val)

    df = pd.DataFrame(results)
    df["sd"] = noise_sd

    # Store failure counts in metadata or print/return separately if needed
    total_failures = sum(failures.values())
    if total_failures > 0:
        print(f"Failures encountered: {failures}")

    return df
