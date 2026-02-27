"""Utils for simple ratio imaging calculation."""

from collections import defaultdict
from typing import cast

import numpy as np
import pandas as pd
import tifffile as tff
from dask.array.core import Array
from numpy.typing import NDArray
from scipy import optimize, stats

from nima.nima import AXES_LENGTH_4D

from .nima_types import ImFrame, ImMask, ImSequence
from .segmentation import _bgmax, calculate_bg_iteratively, prob


# fit the bg for clop3 experiments
def bg(
    im: ImFrame, bgmax: float | None = None
) -> tuple[
    float,
    float,
]:
    """Estimate image bg w/out iteration.

    Parameters
    ----------
    im : ImFrame
        Single YX image.
    bgmax: float | None
        Maximum value for bg?.

    Returns
    -------
    tuple[float, float]
        Background and standard deviation values.

    Examples
    --------
    r = bg(np.ones([10, 10]))
    plt.step(r[2], r[3])

    Notes
    -----
    Faster than `nimg` by 2 order of magnitude.

    """

    def fitfunc(p: NDArray[np.float64], x: NDArray[np.float64]) -> NDArray[np.float64]:
        return cast(
            "NDArray[np.float64]", p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) + p[3]
        )

    def errfunc(
        p: NDArray[np.float64], x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        residuals = y - fitfunc(p, x)
        return residuals.astype(np.float64)

    mmin = int(im.min())
    mmax = int(im.max())
    if bgmax is None:
        bgmax = (mmin + mmax) / 2
    vals = im[im < bgmax]
    ydata, xdata = np.histogram(vals, bins=mmax - mmin, range=(mmin, mmax))
    xdata = xdata[:-1] + 0.5
    loc, scale = stats.distributions.norm.fit(vals)
    init = [sum(ydata), loc, scale, min(ydata)]
    fin = len(xdata) - 1
    leastsq = optimize.leastsq
    out = leastsq(errfunc, init, args=(xdata[:fin], ydata[:fin]))
    return out[0][1], out[0][2]


def ave(img: ImFrame, bgmax: float, prob_value: float = 0.001) -> float:
    """Mask out the bg and return objects average of a frame."""
    if bgmax:
        # MAYBE: Use bg2
        pass
    bg_result = calculate_bg_iteratively(img)
    (
        av,
        sd,
    ) = bg_result.bg, bg_result.sd
    av = min(av, 20)
    sd = min(sd, 10)
    mask = prob(img, float(av), sd) < prob_value
    # MAYBE: plot the mask
    return np.ma.masked_array(img, ~mask).mean() - av  # type: ignore[no-any-return]


def channel_mean(img: ImFrame) -> pd.DataFrame:
    """Average each channel frame by frame."""
    r = defaultdict(list)
    for t in range(img.shape[0]):
        if img.ndim == AXES_LENGTH_4D:
            for c in range(img.shape[1]):
                r[str(c)].append(ave(img[t, c], bgmax=_bgmax(img[t, c])))
        else:
            r["YFP"].append(ave(img[t], bgmax=50))
    return pd.DataFrame(r)


def ratio_df(filelist: list[str]) -> pd.DataFrame:
    """Compute ratios from a list of files."""  # noqa: DOC501
    r = []
    for f in filelist:
        img = tff.imread(f)
        if isinstance(img, np.ndarray):
            if img.dtype in {np.float64, np.int_}:
                r.append(channel_mean(img))
            else:
                msg = (
                    f"Expected an ImArray with dtype np.float_ or np.int_, "
                    f"but received dtype {img.dtype}"
                )
                raise TypeError(msg)
    combined_df = pd.concat(r, ignore_index=True)
    if "YFP" in combined_df:
        combined_df["norm"] = combined_df["YFP"] / combined_df["YFP"][:5].mean()
    else:
        combined_df["r_Cl"] = combined_df[2] / combined_df[1]
        combined_df["r_pH"] = combined_df[0] / combined_df[2]
    return combined_df


def mask_all_channels(
    image: ImSequence | Array, thresholds: tuple[float, ...]
) -> ImMask | Array:
    """Mask a multichannel plane.

    Parameters
    ----------
    image : ImSequence | Array
        CYX multichannel image.
    thresholds : tuple[float, ...]
        threshold values

    Returns
    -------
    ImMask | Array
        Multichannel mask.

    Raises
    ------
    ValueError
        Assertion Error for mismatching number of channels.

    Examples
    --------
    >>> import tifffile
    >>> fp = "tests/data/1b_c16_15.tif"
    >>> dd = tifffile.imread(fp)
    >>> int(mask_all_channels(dd[0, :], [19, 17, 22]).sum())
    262144
    """
    if len(thresholds) != image.shape[0]:
        msg = "Number of thresholds must match the number of channels in the image."
        raise ValueError(msg)

    is_dask_array: bool = isinstance(image, Array)

    # Create the initial mask for the first channel
    mask = image[0] > thresholds[0]

    # Combine conditions for the rest of the channels
    for channel_index, threshold in enumerate(thresholds[1:], 1):
        mask &= image[channel_index] > threshold

    # If it's a Dask array, return as-is unless it's finalized; otherwise, compute
    if is_dask_array:
        return cast("Array", mask)
    return cast("ImMask", np.asarray(mask))
