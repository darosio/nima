"""Utils for simple ratio imaging calculation."""

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd
import tifffile as tff  # type: ignore[import-untyped]
from numpy.typing import NDArray
from scipy import optimize, signal, stats  # type: ignore[import-untyped]

from nima.nima import AXES_LENGTH_4D

from .segmentation import _bgmax, iteratively_refine_background, prob
from .types import ImArray, ImMask


# fit the bg for clop3 experiments
def bg(
    im: ImArray, bgmax: float | None = None
) -> tuple[
    float,
    float,
]:
    """Estimate image bg.

    Parameters
    ----------
    im : ImArray
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

    def fitfunc(
        p: list[float], x: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        return p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) + p[3]

    def errfunc(
        p: list[float], x: float | NDArray[np.float64], y: float | NDArray[np.float64]
    ) -> float | NDArray[np.float64]:
        return y - fitfunc(p, x)

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


def ave(img: NDArray[np.float64], bgmax: float, prob_value: float = 0.001) -> float:
    """Mask out the bg and return objects average of a frame."""
    if bgmax:
        # MAYBE: Use bg2
        pass
    bg_result = iteratively_refine_background(img)
    (
        av,
        sd,
    ) = bg_result.bg, bg_result.sd
    av = min(av, 20)
    sd = min(sd, 10)
    mask = prob(img, float(av), sd) < prob_value
    # MAYBE: plot the mask
    return np.ma.masked_array(img, ~mask).mean() - av  # type: ignore[no-untyped-call, no-any-return]


def channel_mean(img: ImArray) -> pd.DataFrame:
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
    """Compute ratios from a list of files."""
    r = []
    for f in filelist:
        img = tff.imread(f)
        if isinstance(img, np.ndarray):
            if img.dtype in (np.float64, np.int_):
                r.append(channel_mean(img))
            else:
                msg = (
                    f"Expected an ImArray with dtype np.float_ or np.int_, "
                    f"but received dtype {img.dtype}"
                )
                raise TypeError(msg)
        else:
            msg = f"Expected an ImArray, but received {type(img)}"
            raise TypeError(msg)
    combined_df = pd.concat(r, ignore_index=True)
    if "YFP" in combined_df:
        combined_df["norm"] = combined_df["YFP"] / combined_df["YFP"][:5].mean()
    else:
        combined_df["r_Cl"] = combined_df[2] / combined_df[1]
        combined_df["r_pH"] = combined_df[0] / combined_df[2]
    return combined_df


def bg2(
    img: ImArray, step: float = 0.2, bgmax: None | float = None
) -> tuple[float, float, NDArray[np.signedinteger[Any]], NDArray[np.floating[Any]]]:
    """Estimate image bg."""
    if bgmax is None:
        bgmax = _bgmax(img)
    vals = img[img < bgmax]
    mmin = vals.min()
    mmax = vals.max()
    density = stats.gaussian_kde(vals)
    x = np.arange(mmin, mmax, step=step)
    density = density(x)
    # MAYBE: plot x, density
    pos_max = signal.find_peaks(density, width=2, rel_height=0.1)[0][0]
    v = density[pos_max] / 2
    pos_delta = signal.find_peaks(-np.absolute(density - v), width=2, rel_height=0.2)[
        0
    ][0]
    delta = (pos_max - pos_delta) * step
    return pos_max * step + mmin, delta, x, density


def mask_all_channels(im: ImArray, thresholds: tuple[float]) -> ImMask:
    """Mask a multichannel plane.

    Parameters
    ----------
    im : ImArray
        CYX multichannel image.
    thresholds : tuple[float]
        threshold values

    Returns
    -------
    ImMask
        Multichannel mask.

    Raises
    ------
    ValueError
        Assertion Error for mismatching number of channels.

    Examples
    --------
    >>> from aicsimageio import AICSImage, readers
    >>> reader = readers.tiff_reader.TiffReader
    >>> fp = "tests/data/1b_c16_15.tif"
    >>> ir = AICSImage(fp, reader=reader)
    >>> dd = ir.dask_data
    >>> mask_all_channels(dd[0, :, 0], [19, 17, 22]).compute().sum()
    262144
    """
    if len(thresholds) != im.shape[0]:
        msg = "Length of thresholds must match the number of image dimensions."
        raise ValueError(msg)
    m: ImMask = im[0] > thresholds[0]
    thr: float
    for i, thr in enumerate(thresholds[1:], 1):
        m = m & (im[i] > thr)
    return m
