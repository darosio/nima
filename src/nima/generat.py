"""Generate mock images."""

import random
import warnings
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def gen_bias(nrows: int = 128, ncols: int = 128) -> NDArray[np.float64]:
    """Generate a bias frame."""
    xvec = np.arange(ncols)
    yvec = 2 - (xvec**2 / 2.6 * (np.sin(xvec / 20) ** 2 + 0.1)) / 4000
    img = np.tile(yvec * 2, (nrows, 1))
    return img


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
    nrows: int = 128, ncols: int = 128, min_radius: int = 6, max_radius: int = 12
) -> NDArray[np.bool_]:
    """Generate a single ellipsoid object with random shape and position."""
    # Inspired by http://scipy-lectures.org/packages/scikit-image/index.html.
    x_idx, y_idx = np.indices((nrows, ncols))
    x_obj, y_obj = np.random.randint(nrows), np.random.randint(ncols)
    radius = np.random.randint(min_radius, max_radius)
    ellipsis = np.random.rand() * 3.5 - 1.75
    mask = (
        (x_idx - x_obj) ** 2
        + (y_idx - y_obj) ** 2
        + ellipsis * (x_idx - x_obj) * (y_idx - y_obj)
        < radius**2
    ).astype(np.bool_)
    return mask  # type: ignore[no-any-return]


@dataclass
class ImageObjsParams:
    """Parameters for an image frame."""

    max_num_objects: int = 8
    nrows: int = 128
    ncols: int = 128
    min_radius: int = 6
    max_radius: int = 12
    max_fluor: float = 20.0


def gen_objs(params: ImageObjsParams | None = None) -> NDArray[np.float64]:
    """Generate a frame with ellipsoid objects; random n, shape, position and I."""
    params = ImageObjsParams() if params is None else params
    min_num_objects = 2
    num_objs = (
        random.randint(min_num_objects, params.max_num_objects)  # nosec "no-secure-random"
        if params.max_num_objects > min_num_objects
        else params.max_num_objects
    )
    # MAYBE: convolve the obj to simulate lower peri-cellular profile
    objs = [
        params.max_fluor
        * np.random.rand()
        * gen_object(params.nrows, params.ncols, params.min_radius, params.max_radius)
        for _ in range(num_objs)
    ]
    img = np.sum(objs, axis=0)
    return img  # type: ignore[no-any-return]


def gen_frame(
    objs: NDArray[np.float64],
    bias: NDArray[np.float64] | None = None,
    flat: NDArray[np.float64] | None = None,
    sky: float = 2,
    noise_sd: float = 1,
) -> NDArray[np.float64]:  # pylint: disable=too-many-arguments
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
    noise = np.random.normal(0, noise_sd, size=(nrows, ncols))
    img = bias + flat * (sky + objs) + noise
    return img.clip(0).astype("uint16")
