"""Generate mock images."""

from __future__ import annotations

import warnings

import numpy as np
import numpy.typing as npt


def gen_bias(nrows: int = 128, ncols: int = 128) -> npt.NDArray[np.float_]:
    """Generate a bias frame."""
    xvec = np.array(list(range(ncols)))
    yvec = 2 - (xvec**2 / 2.6 * ((np.sin(xvec / 20)) ** 2 + 0.1)) / 4000
    img = np.ones((nrows, ncols))
    for i in range(nrows):
        img[i, :] = yvec * 2
    return img


def gen_flat(nrows: int = 128, ncols: int = 128) -> npt.NDArray[np.float_]:
    """Generate a flat frame."""
    img = np.ones((nrows, ncols))

    def zfunc(xpos: int, ypos: int) -> float:
        zval = (
            -0.0001 * xpos**2
            - 0.0001 * ypos**2
            + 0.016 * ypos
            + 0.014 * xpos
            + 0.5
            + 0.000015 * xpos * ypos
        )
        return zval

    for i in range(nrows):
        for j in range(ncols):
            img[i, j] = zfunc(j, i)
    img /= img.mean()
    return img


def gen_object(
    nrows: int = 128, ncols: int = 128, min_radius: int = 6, max_radius: int = 12
) -> npt.NDArray[np.bool_]:
    """Mimic http://scipy-lectures.org/packages/scikit-image/index.html."""
    x_idx, y_idx = np.indices((nrows, ncols))
    x_obj = np.random.randint(nrows)
    y_obj = np.random.randint(ncols)
    radius = np.random.randint(min_radius, max_radius)
    ellipsis = np.random.rand() * 3.5 - 1.75
    mask = np.array(
        (x_idx - x_obj) ** 2
        + (y_idx - y_obj) ** 2
        + ellipsis * (x_idx - x_obj) * (y_idx - y_obj)
        < radius**2
    )
    return mask


def gen_objs(
    max_fluor: float = 20, max_n_obj: int = 8, **kwargs: int
) -> npt.NDArray[np.float_]:
    """Generate a frame with ellipsoid objects; in random number, shape, position and intensity."""
    img = max_fluor * np.random.rand() * gen_object(**kwargs)
    # MAYBE: convolve the obj to simulate lower peri-cellular profile
    for _ in range(1, np.random.randint(2, max_n_obj)):
        img += max_fluor * np.random.rand() * gen_object(**kwargs)
    return img


def gen_frame(
    objs: npt.NDArray[np.float_],
    bias: npt.NDArray[np.float_] | None = None,
    flat: npt.NDArray[np.float_] | None = None,
    dark: float = 0,
    sky: float = 2,
    noise_sd: float = 1,
) -> npt.NDArray[np.float_]:  # pylint: disable=too-many-arguments
    """Simulate an acquired frame [bias + noise + dark + flat * (sky + obj)]."""
    (nrows, ncols) = objs.shape
    if bias is None:
        bias = gen_bias(nrows, ncols)
    elif bias.shape != (nrows, ncols):
        warnings.warn("Shape mismatch. Generate Bias...", UserWarning)
        bias = gen_bias(nrows, ncols)
    if flat is None:
        flat = gen_flat(nrows, ncols)
    elif flat.shape != (nrows, ncols):
        warnings.warn("Shape mismatch. Generate Flat...", UserWarning)
        flat = gen_flat(nrows, ncols)
    noise = np.random.normal(0, noise_sd, size=(nrows, ncols))
    img = bias + flat * (sky + objs) + dark + noise
    return img  # type: ignore
