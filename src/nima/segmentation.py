"""Functions to partition images into meaningful regions."""

from typing import Any, cast, overload

import matplotlib.pyplot as plt
import numpy as np
import skimage
from numpy.typing import NDArray
from scipy import ndimage, optimize, signal, special, stats  # type: ignore
from skimage import filters, morphology

from .types import ImArray

# TODO: add new bg/fg segmentation based on conditional probability but
# working with dask arrays. Try being clean: define function only for NDArray
# then map dask to use it somehow.


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
            lim = filters.rank.entropy(im8 / im8.max(), morphology.disk(radius))  # type: ignore
        else:
            lim = filters.rank.entropy(im8, morphology.disk(radius))  # type: ignore
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


@overload
def prob(v: float, bg: float, sd: float) -> float: ...
@overload
def prob(v: ImArray, bg: float, sd: float) -> NDArray[np.float_]: ...


def prob(v: float | ImArray, bg: float, sd: float) -> float | NDArray[np.float_]:
    """Compute pixel probability of belonging to background."""
    result = special.erfc((v - bg) / sd)
    # Use typing.cast to explicitly inform mypy
    if isinstance(v, float):
        return cast(float, result)
    else:
        return cast(NDArray[np.float_], result)


def fit_gaussian(vals: NDArray[np.float_]) -> tuple[float, float]:
    """Estimate mean and standard deviation using a Gaussian fit.

        The function fits a Gaussian distribution to a given array of values and estimates
    the mean and standard deviation of the distribution. This process involves constructing
    a histogram of the input values, fitting the Gaussian model to the histogram, and
    optimizing the parameters of the Gaussian function to best match the data.

    Parameters
    ----------
    vals : NDArray[np.float_]
        A one-dimensional NumPy array containing the data values for which the Gaussian
        distribution parameters (mean and standard deviation) are to be estimated.

    Returns
    -------
    mean : float
        Estimated mean (mu) of the Gaussian distribution.
    sd : float
        Estimated standard deviation (sigma) of the Gaussian distribution.

    Examples
    --------
    >>> import numpy as np
    >>> from numpy.random import default_rng
    >>> rng = default_rng()
    >>> data = rng.normal(loc=50, scale=5, size=1000)  # Generate sample data
    >>> mean, sd = fit_gaussian(data)
    >>> print(f"Estimated Mean: {mean}, Estimated Standard Deviation: {sd}")

    Notes
    -----
    The Gaussian fitting process involves constructing a histogram from the input
    array and then fitting a Gaussian model to this histogram. The optimization is performed
    using the least squares method to minimize the difference between the histogram of the data
    and the Gaussian function defined as:

        f(x) = amplitude * exp(-0.5 * ((x - mean) / sigma)^2) + offset

    where amplitude, mean, sigma, and offset are parameters of the Gaussian function, with
    'mean' and 'sigma' being the primary parameters of interest in this function.

    This function relies on the `leastsq` optimization function from `scipy.optimize` and
    the method `norm.fit` from `scipy.stats.distributions` to estimate initial parameters
    for the optimization process.
    """

    def gaussian_fit_func(
        params: list[float], x: float | NDArray[np.float_]
    ) -> NDArray[np.float_]:
        amplitude, mean, sigma, offset = params
        return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset

    def fit_error_func(
        params: list[float], x: NDArray[np.float_], y: NDArray[np.float_]
    ) -> NDArray[np.float_]:
        return y - gaussian_fit_func(params, x)

    min_val, max_val = int(vals.min()), int(vals.max())
    ydata, edges = np.histogram(vals, bins=max_val - min_val, range=(min_val, max_val))
    xdata = edges[:-1] + 0.5
    initial_guess = [ydata.sum(), *stats.distributions.norm.fit(vals), ydata.min()]
    optimized_params = optimize.leastsq(
        fit_error_func, initial_guess, args=(xdata[:-1], ydata[:-1])
    )[0]
    mean, sd = optimized_params[1], optimized_params[2]
    return mean, sd


# fit the bg for clop3 experiments
def iteratively_refine_background(
    frame: NDArray[np.float_], bgmax: None | np.float_ = None, probplot: bool = False
) -> tuple[float, float, None | tuple[float, float, float], None | plt.Figure]:
    """Iteratively refines the background estimate of an image frame using Gaussian fitting.

    This function takes a single image frame, performs an initial estimate of the background
    using the median value, and then iteratively refines this estimate by applying a Gaussian
    fit on values beneath this background level. The process is repeated until convergence is
    achieved, enhancing the accuracy of the background estimate.

    Parameters
    ----------
    frame : NDArray[np.float_]
        The image frame for which the background estimate needs to be refined.
    bgmax : None | np.float_, optional
        Maximum value used from `frame` for background estimation. Defaults to
        None, using the mean of all pixels.
    probplot : bool, optional
        If True, generates a Q-Q plot to assess Gaussian fit. Default is False.

    Returns
    -------
    bg_final : float
        The refined background estimate after convergence.
    sd_final : float
        The standard deviation of the Gaussian fit corresponding to the final
        background estimate.
    probplot_fit : None | tuple[float, float, float]
        Tuple containing probplot parameters: slope, intercept, and R-value of
        the probability plot (Q-Q plot) if `probplot` is True; otherwise, None.
    probplot_fig : None | plt.Figure
        The figure object of the probability plot if `probplot` is True;
        otherwise, None.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> frame = np.random.normal(loc=100, scale=10, size=(256, 256))
    >>> bg_final, sd_final = iteratively_refine_background(frame)
    >>> print(f"Refined Background: {bg_final}, Standard Deviation: {sd_final}")
    """
    # Initial background estimate using the median of the frame
    bg_max = 1.5 * np.mean(frame) if bgmax is None else bgmax
    vals_below_bg_max = frame[frame < bg_max]
    bg_initial, sd_initial = fit_gaussian(vals_below_bg_max)
    # Iterative refinement
    bg_final = bg_initial
    for i in range(100):  # Maximum of 100 iterations for refinement
        # Filtering using the current background estimate
        prob_frame = prob(frame, bg_final, sd_initial)
        mask = ndimage.percentile_filter(prob_frame, percentile=1, size=2) > 0.005
        filtered_frame = frame[mask]
        bg_updated, sd_updated = fit_gaussian(filtered_frame)
        if np.isclose(bg_updated, bg_final, atol=1e-6):  # Tolerance for convergence
            break
        bg_final = bg_updated
    # Return also a probability plot
    if probplot:
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        ax1.hist(vals_below_bg_max, bins=20)
        ax2 = fig.add_subplot(132)
        ax2.set_title("Probplot")
        _, fit = stats.probplot(vals_below_bg_max, plot=ax2, rvalue=True)
        ax3 = fig.add_subplot(133)
        masked = frame * mask
        img0 = ax3.imshow(masked)
        plt.colorbar(img0, ax=ax3, orientation="horizontal")  # type: ignore
        fig.tight_layout()
    else:
        fit, fig = None, None
    return bg_final, sd_updated, fit, fig
