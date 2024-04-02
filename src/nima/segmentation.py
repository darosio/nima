"""Functions to partition images into meaningful regions."""

from collections import defaultdict
from typing import Any, NewType, TypeVar, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import tifffile as tff  # type: ignore
from dask.array.core import Array as DaskArray
from dask.diagnostics.progress import ProgressBar
from numpy.typing import NDArray
from scipy import ndimage, optimize, signal, special, stats  # type: ignore

from nima import utils

# TODO: add new bg/fg segmentation based on conditional probability but
# working with dask arrays. Try being clean: define function only for NDArray
# then map dask to use it somehow.

ImArray = TypeVar("ImArray", NDArray[np.float_], NDArray[np.int_])
ImMask = NewType("ImMask", NDArray[np.bool_])


def myhist(
    im: ImArray,
    bins: int = 60,
    log: bool = False,
    nf: bool = False,
) -> None:
    """Plot image intensity as histogram.

    ..note:: Consider deprecation.

    """
    hist, bin_edges = np.histogram(im, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if nf:
        plt.figure()
    plt.plot(bin_centers, hist, lw=2)
    if log:
        plt.yscale("log")  # type: ignore


def bg(  # noqa: C901
    im: ImArray,
    kind: str = "arcsinh",
    perc: float = 10.0,
    radius: int | None = 10,
    adaptive_radius: int | None = None,
    arcsinh_perc: int | None = 80,
) -> tuple[float, NDArray[np.int_] | NDArray[np.float_], list[plt.Figure]]:
    """Bg segmentation.

    Return median, whole vector, figures (in a [list])

    Parameters
    ----------
    im: ImArray
        An image stack.
    kind : str, optional
        Method {'arcsinh', 'entropy', 'adaptive', 'li_adaptive', 'li_li'} used for the
        segmentation.
    perc : float, optional
        Perc % of max-min (default=10) for thresholding *entropy* and *arcsinh*
        methods.
    radius : int | None, optional
        Radius (default=10) used in *entropy* and *arcsinh* (percentile_filter)
        methods.
    adaptive_radius : int | None, optional
        Size for the adaptive filter of skimage (default is im.shape[1]/2).
    arcsinh_perc : int | None, optional
        Perc (default=80) used in the percentile_filter (scipy) within
        *arcsinh* method.

    Returns
    -------
    median : float
        Median of the bg masked pixels.
    pixel_values : NDArray[np.int_] | NDArray[np.float_]
        Values of all bg masked pixels.
    figs : list[plt.Figure]
        List of fig(s). Only entropy and arcsinh methods have 2 elements.

    Raises
    ------
    Exception
        When % radius is out of bounds.

    """
    if adaptive_radius is None:
        adaptive_radius = int(im.shape[1] / 2)
        if adaptive_radius % 2 == 0:  # sk >0.12.0 check for even value
            adaptive_radius += 1
    min_perc, max_perc = 0.0, 100.0
    if (perc < min_perc) or (perc > max_perc):
        raise Exception("perc must be in [0, 100] range")
    else:
        perc /= 100
    lim_ = False
    m = np.ones_like(im)  # default value for m; instead of m = None
    if kind == "arcsinh":
        lim = np.arcsinh(im)
        lim = ndimage.percentile_filter(lim, arcsinh_perc, size=radius)
        lim_ = True
        title: Any = radius, perc
        thr = (1 - perc) * lim.min() + perc * lim.max()
        m = lim < thr
    elif kind == "entropy":
        im8 = skimage.util.img_as_ubyte(im)  # type: ignore
        if im.dtype == float:
            lim = filters.rank.entropy(im8 / im8.max(), disk(radius))  # type: ignore
        else:
            lim = filters.rank.entropy(im8, disk(radius))  # type: ignore
        lim_ = True
        title = radius, perc
        thr = (1 - perc) * lim.min() + perc * lim.max()
        m = lim < thr
    elif kind == "adaptive":
        lim_ = False
        title = adaptive_radius
        f = im > filters.threshold_local(im, adaptive_radius)  # type: ignore
        m = ~f
    elif kind == "li_adaptive":
        lim_ = False
        title = adaptive_radius
        li = filters.threshold_li(im.copy())  # type: ignore
        m = im < li
        # # FIXME: in case m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        f = imm > filters.threshold_local(imm, adaptive_radius)  # type: ignore
        m = ~f * m
    elif kind == "li_li":
        lim_ = False
        title = None
        li = filters.threshold_li(im.copy())  # type: ignore
        m = im < li
        # # FIXME: in case m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        # To avoid zeros generated after first thesholding, clipping to the
        # min value of original image is needed before second thesholding.
        thr2 = filters.threshold_li(imm.clip(np.min(im)))  # type: ignore
        m = im < thr2
        # # FIXME: in case mm = skimage.morphology.binary_closing(mm)
    elif kind == "inverse_local_yen":
        title = None
        f = filters.threshold_local(1 / im)  # type: ignore
        m = f > filters.threshold_yen(f)  # type: ignore
    pixel_values = im[m]
    iqr = np.percentile(pixel_values, [25, 50, 75])

    def plot() -> plt.Figure:
        f = plt.figure(figsize=(9, 5))
        ax1 = f.add_subplot(121)
        masked = im * m
        cmap = plt.cm.inferno  # type: ignore
        img0 = ax1.imshow(masked, cmap=cmap)
        plt.colorbar(img0, ax=ax1, orientation="horizontal")  # type: ignore
        plt.title(kind + " " + str(title) + "\n" + str(iqr))
        f.add_subplot(122)
        myhist(im[m], log=True)
        f.tight_layout()
        return f

    f1 = plot()
    figures = [f1]
    if lim_:

        def plot_lim() -> plt.Figure:
            f = plt.figure(figsize=(9, 4))
            ax1, ax2, host = f.subplots(nrows=1, ncols=3)  # type: ignore
            img0 = ax1.imshow(lim)
            plt.colorbar(img0, ax=ax2, orientation="horizontal")  # type: ignore
            # FIXME: this is horribly duplicating an axes
            f.add_subplot(132)
            myhist(lim)
            #
            # plot bg vs. perc
            ave, sd, median = ([], [], [])
            delta = lim.max() - lim.min()
            delta /= 2
            rng = np.linspace(lim.min() + delta / 20, lim.min() + delta, 20)
            par = host.twiny()
            # Second, show the right spine.
            par.spines["bottom"].set_visible(True)
            par.set_xlabel("perc")
            par.set_xlim(0, 0.5)
            par.grid()
            host.set_xlim(lim.min(), lim.min() + delta)
            p = np.linspace(0.025, 0.5, 20)
            for t in rng:
                m = lim < t
                ave.append(im[m].mean())
                sd.append(im[m].std() / 10)
                median.append(np.median(im[m]))
            host.plot(rng, median, "o")
            par.errorbar(p, ave, sd)
            f.tight_layout()
            return f

        f2 = plot_lim()
        figures.append(f2)
    # Close all figures explicitly just before returning
    for fig in figures:
        plt.close(fig)

    return iqr[1], pixel_values, figures


def _bgmax(img: ImArray, step: int = 4) -> float:
    thr = skimage.filters.threshold_mean(img)  # type: ignore
    vals = img[img < thr / 1]
    mmin: float = vals.min()
    mmax = vals.max()
    density = stats.gaussian_kde(vals)(
        np.linspace(mmin, mmax, num=((mmax - mmin) // step))
    )
    # fail with G550E_CFTR_DMSO_1
    peaks_indices = signal.find_peaks(-density, width=2, rel_height=0.1)[0]
    if peaks_indices.size > 0:
        first_peak_val = peaks_indices[0]
        result = mmin + (first_peak_val * step)
        return float(result)
    else:
        # Handle the case where no peaks are found
        return mmin


# fit the bg for clop3 experiments
def bgnima(im: ImArray, bgmax: float | None = None) -> tuple[
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
        p: list[float], x: float | NDArray[np.float_]
    ) -> float | NDArray[np.float_]:
        return p[0] * np.exp(-0.5 * ((x - p[1]) / p[2]) ** 2) + p[3]

    def errfunc(
        p: list[float],
        x: float | NDArray[np.float_],
        y: float | NDArray[np.float_],
    ) -> float | NDArray[np.float_]:
        return y - fitfunc(p, x)

    mmin = int(im.min())
    mmax = int(im.max())
    if bgmax is None:
        # after 240315_IC/G550E-R1070W_VX809_FRK1.tf8: bgmax = (mmin + mmax) / 2
        bgmax = im.mean()
    vals = im[im < bgmax]
    ydata, xdata = np.histogram(vals, bins=mmax - mmin, range=(mmin, mmax))
    xdata = xdata[:-1] + 0.5
    loc, scale = stats.distributions.norm.fit(vals)
    init = [sum(ydata), loc, scale, min(ydata)]
    fin = len(xdata) - 1
    leastsq = optimize.leastsq
    out = leastsq(errfunc, init, args=(xdata[:fin], ydata[:fin]))
    bg, sd = out[0][1], out[0][2]
    # mgeo = ndimage.gaussian_filter(utils.prob(im, bg, sd), 0.25) > 0.005
    mgeo = (
        ndimage.percentile_filter(utils.prob(im, bg, sd), percentile=1, size=2) > 0.005
    )
    # pixel_values = im[im < bg + 1.67 * sd]
    pixel_values = im[mgeo]
    return bg, sd, pixel_values


def bgnima2(
    img: ImArray, step: float = 0.2, bgmax: float = 60.0
) -> tuple[float, float, NDArray[np.signedinteger[Any]], NDArray[np.floating[Any]]]:
    """Estimate image bg."""
    bgmax = _bgmax(img) if bgmax is None else bgmax
    values_under_bgmax = img[img < bgmax]
    mmin, mmax = values_under_bgmax.min(), values_under_bgmax.max()
    x = np.arange(mmin, mmax, step=step)
    density = stats.gaussian_kde(values_under_bgmax)(x)
    # MAYBE: plot x, density
    pos_max = signal.find_peaks(density, width=2, rel_height=0.1)[0][0]
    v = density[pos_max] / 2
    pos_delta = signal.find_peaks(-np.absolute(density - v), width=2, rel_height=0.2)[
        0
    ][0]
    delta = (pos_max - pos_delta) * step
    return pos_max * step + mmin, delta, x, density


# TODO: Contradicting the use of NDArray only. To experiment into dask image...
ImArray2 = TypeVar("ImArray", NDArray[np.float_], NDArray[np.int_], DaskArray)


def bgnima2_dask(
    img: ImArray, step: float = 0.3, bgmax: float = 60.0
) -> tuple[float, float, np.ndarray, np.ndarray]:
    def compute_bgmax(img: ImArray, bgmax=None) -> float:
        if bgmax is not None:
            return bgmax
        else:
            # Assuming _bgmax function is compatible with Dask arrays or is replaced with a suitable alternative
            return float(img.max().compute())

    def compute_kde_peaks(
        values_under_bgmax: np.ndarray, mmin: float, mmax: float, step: float
    ):
        x = np.arange(mmin, mmax, step=step)
        density = stats.gaussian_kde(values_under_bgmax)(x)
        pos_max = signal.find_peaks(density, width=2, rel_height=0.1)[0][0]
        v = density[pos_max] / 2
        pos_delta = signal.find_peaks(
            -np.absolute(density - v), width=2, rel_height=0.2
        )[0][0]
        delta = (pos_max - pos_delta) * step
        return pos_max * step + mmin, delta, x, density

    bgmax = compute_bgmax(img, bgmax)
    values_under_bgmax = img[img < bgmax].compute()
    mmin, mmax = values_under_bgmax.min(), values_under_bgmax.max()

    pos_max, delta, x, density = compute_kde_peaks(values_under_bgmax, mmin, mmax, step)

    return pos_max, delta, x, density


def bgnima3_stack(
    img_stack: ImArray, step: float = 0.3, bgmax: float = 60.0
) -> list[tuple[float, float, np.ndarray, np.ndarray]]:
    def compute_bgmax(img: ImArray, bgmax=None) -> float:
        if bgmax is not None:
            return bgmax
        else:
            return float(img.max().compute())

    def compute_kde_peaks(
        values_under_bgmax: np.ndarray, mmin: float, mmax: float, step: float
    ):
        x = np.arange(mmin, mmax, step=step)
        density = stats.gaussian_kde(values_under_bgmax)(x)
        pos_max = signal.find_peaks(density, width=2, rel_height=0.1)[0][0]
        v = density[pos_max] / 2
        pos_delta = signal.find_peaks(
            -np.absolute(density - v), width=2, rel_height=0.2
        )[0][0]
        delta = (pos_max - pos_delta) * step
        return pos_max * step + mmin, delta, x, density

    results = []
    for img in img_stack:
        bgmax_val = compute_bgmax(img, bgmax)
        values_under_bgmax = img[img < bgmax_val].compute()
        mmin, mmax = float(values_under_bgmax.min()), float(values_under_bgmax.max())

        result = compute_kde_peaks(values_under_bgmax, mmin, mmax, step)
        results.append(result)

    return results
