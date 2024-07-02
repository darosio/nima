"""Functions to partition images into meaningful regions."""

from dataclasses import dataclass
from typing import cast, overload

import matplotlib.pyplot as plt
import numpy as np
import skimage
from dask.diagnostics.progress import ProgressBar
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import (  # type: ignore[import-untyped]
    ndimage,
    optimize,
    signal,
    special,
    stats,
)
from skimage import filters, morphology

from .types import ImArray, ImMask

# TODO: add new bg/fg segmentation based on conditional probability but
# working with dask arrays. Try being clean: define function only for NDArray
# then map dask to use it somehow.


def _bg_plot(im: ImArray, m: ImMask, title: str, lim: ImArray | None) -> list[Figure]:
    fig1 = plt.figure(figsize=(9, 5))
    ax1 = fig1.add_subplot(121)
    masked = im * m
    cmap = plt.get_cmap("inferno")
    img0 = ax1.imshow(masked, cmap=cmap)
    plt.colorbar(img0, ax=ax1, orientation="horizontal")
    plt.title(title)
    fig1.add_subplot(122)
    plt.hist(im[m].ravel(), bins=60, log=True, color="skyblue", edgecolor="black")
    plt.title("Histogram of pixels considered to be background")
    fig1.tight_layout()
    figures = [fig1]
    if lim is not None:
        fig2 = plt.figure(figsize=(9, 4))
        ax1, ax2, host = fig2.subplots(nrows=1, ncols=3)  # type: ignore[misc]
        img0 = ax1.imshow(lim)
        plt.colorbar(img0, ax=ax2, orientation="horizontal")
        ax2.hist(lim.ravel(), bins=60, log=True, color="lightgray", edgecolor="black")
        ax2.set_title("Histogram of Lim")
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
        fig2.tight_layout()
        figures.append(fig2)
    # Close all figures explicitly just before returning
    for fig in figures:
        plt.close(fig)
    return figures


# MAYBE: Convert to comments like #: Attribute desc
@dataclass
class BgParams:
    """Parameters for `bg` segmentation methods.

    Attributes
    ----------
    kind : str, optional
        The segmentation method to use. Available options are 'arcsinh',
        'entropy', 'adaptive', 'li_adaptive', 'li_li', and 'inverse_yen'.
        The default is 'arcsinh'.
    perc : float, optional
        Percentage of max-min (default=10) for threshold `entropy` and `arcsinh`
        methods. It gets automatically converted to a fraction of 1 in the
        initialization process.
    radius : int, optional
        Radius (default=10) used in `entropy` and `arcsinh` (percentile_filter)
        methods.
    adaptive_radius : int | None, optional
        Size for the `adaptive` filter of skimage (default is calculated as
        half of the image size during usage).
    arcsinh_perc : float, optional
        Percentage (default=80) used in the percentile_filter (scipy) within
        `arcsinh` method.
    erosion_disk : float, optional
        Size of the erosion disk used during image processing (default: 0).
    clip : bool, optional
        Flag to enable or disable clipping (default: False).

    Raises
    ------
    ValueError
        If `perc` is not within the [0, 100] range.
    """

    kind: str = "arcsinh"
    perc: float = 10.0
    radius: int = 10
    adaptive_radius: int | None = None
    arcsinh_perc: float = 80
    erosion_disk: float = 0
    clip: bool = False

    def __post_init__(self) -> None:
        """Perform validation and normalization on initialization.

        Ensures that `perc` is within the valid range and converts it from
        a percentage to a fraction for internal use.

        Raises
        ------
        ValueError
            If `perc` is not within the [0, 100] range.
        """
        min_perc, max_perc = 0.0, 100.0
        if not min_perc <= self.perc <= max_perc:
            msg = "perc must be in [0, 100] range"
            raise ValueError(msg)
        self.perc /= 100


@dataclass
class BgResult:
    """Group result of a frame bg estimation.

    Attributes
    ----------
    bg : float
        Background value estimated from the frame.
    sd : float
        Standard deviation of the background estimation.
    iqr : tuple[float, float, float]
        Interquartile range (IQR) of the background estimation, consisting of
        three float values.
    figures : list[Figure] | None
        List of matplotlib Figure objects representing visualizations related
        to the background estimation, or None if no figures are provided.
    """

    bg: float
    sd: float
    iqr: tuple[float, float, float]
    figures: list[Figure] | None


def _bg_arcsinh(im: ImArray, bg_params: BgParams) -> tuple[ImMask, str, ImArray]:
    perc = bg_params.perc
    radius = bg_params.radius
    arcsinh_perc = bg_params.arcsinh_perc
    lim = np.arcsinh(im)
    lim = ndimage.percentile_filter(lim, arcsinh_perc, size=radius)
    thr = (1 - perc) * lim.min() + perc * lim.max()
    m = lim < thr
    title = (
        f"{bg_params.kind} radius={radius}, perc={perc}, arcsinh_perc={arcsinh_perc}"
    )
    return m, title, lim


def _bg_entropy(im: ImArray, bg_params: BgParams) -> tuple[ImMask, str, ImArray]:
    perc = bg_params.perc
    radius = bg_params.radius
    im8 = skimage.util.img_as_ubyte(im)  # type: ignore[no-untyped-call]
    if im.dtype == float:
        lim = filters.rank.entropy(im8 / im8.max(), morphology.disk(radius))  # type: ignore[no-untyped-call]
    else:
        lim = filters.rank.entropy(im8, morphology.disk(radius))  # type: ignore[no-untyped-call]
    thr = (1 - perc) * lim.min() + perc * lim.max()
    m = lim < thr
    title = f"{bg_params.kind} radius={radius}, perc={perc}"
    return m, title, lim


def _bg_adaptive(im: ImArray, bg_params: BgParams) -> tuple[ImMask, str, None]:
    adaptive_radius = bg_params.adaptive_radius
    f = im > filters.threshold_local(im, adaptive_radius)  # type: ignore[no-untyped-call]
    m = ~f
    title = f"{bg_params.kind} adaptive_radius={adaptive_radius}"
    return m, title, None


def _bg_li_adaptive(im: ImArray, bg_params: BgParams) -> tuple[ImMask, str, None]:
    adaptive_radius = bg_params.adaptive_radius
    li = filters.threshold_li(im)  # type: ignore[no-untyped-call]
    m = im < li
    if bg_params.erosion_disk:
        m = skimage.morphology.binary_erosion(
            m,
            morphology.disk(bg_params.erosion_disk),  # type: ignore[no-untyped-call]
        )
    imm = im * m
    if bg_params.clip:
        imm = imm.clip(np.min(im))
    f = imm > filters.threshold_local(imm, adaptive_radius)  # type: ignore[no-untyped-call]
    m = ~f * m
    title = f"{bg_params.kind} adaptive_radius={adaptive_radius}"
    return m, title, None


def _bg_li_li(im: ImArray, bg_params: BgParams) -> tuple[ImMask, str, None]:
    li = filters.threshold_li(im.copy())  # type: ignore[no-untyped-call]
    m = im < li
    if bg_params.erosion_disk:
        m = skimage.morphology.binary_erosion(
            m,
            morphology.disk(bg_params.erosion_disk),  # type: ignore[no-untyped-call]
        )
    imm = im * m
    # To avoid zeros generated after first thesholding, clipping to the
    # min value of original image is needed before second threshold.
    thr2 = filters.threshold_li(imm.clip(np.min(im)))  # type: ignore[no-untyped-call]
    m = im < thr2
    title = bg_params.kind + " " + ""
    return m, title, None


def _bg_inverse_yen(im: ImArray, bg_params: BgParams) -> tuple[ImMask, str, None]:
    f = filters.threshold_local(1 / im)  # type: ignore[no-untyped-call]
    m = f > filters.threshold_yen(f)  # type: ignore[no-untyped-call]
    title = bg_params.kind + " " + ""
    return m, title, None


# def bg(im: ImArray, bg_params: BgParams | None = None) -> tuple[float, list[Figure]]:
def calculate_bg(im: ImArray, bg_params: BgParams | None = None) -> BgResult:
    """Segment background from an image stack.

    Parameters
    ----------
    im: ImArray
        An image stack.
    bg_params : BgParams | None, optional
        An instance of BgParams containing the parameters for the segmentation.
        If None, default values are used. Default is None.

    Returns
    -------
    BgResult
        Comprise the values for bg, sd and iqr, and a list of matplotlib Figure
        objects related to the segmentation visualization. The actual contents
        depend on the `kind` parameter. For 'entropy' and 'arcsinh' methods,
        this list contains 2 elements.

    Raises
    ------
    ValueError
        If an unsupported segmentation method (kind) is specified.
    """
    bg_params = bg_params if bg_params else BgParams()
    if bg_params.adaptive_radius is None:
        bg_params.adaptive_radius = int(im.shape[1] / 2)
        if bg_params.adaptive_radius % 2 == 0:  # sk >0.12.0 check for even value
            bg_params.adaptive_radius += 1

    processing_functions = {
        "arcsinh": _bg_arcsinh,
        "entropy": _bg_entropy,
        "adaptive": _bg_adaptive,
        "li_adaptive": _bg_li_adaptive,
        "li_li": _bg_li_li,
        "inverse_yen": _bg_inverse_yen,
    }
    if bg_params.kind not in processing_functions:
        msg = f"Invalid 'kind' parameter: {bg_params.kind}"
        raise ValueError(msg)

    m, title, lim = processing_functions[bg_params.kind](im, bg_params)
    pixel_values = im[m]
    iqr = np.percentile(pixel_values, [25, 50, 75])
    title = title + "\n" + str(iqr)
    figures = _bg_plot(im, m, title, lim)
    bg, sd = stats.distributions.norm.fit(pixel_values)
    return BgResult(bg, sd, iqr, figures)


def _bgmax(img: ImArray, bins: int = 50, *, densityplot: bool = False) -> float:
    thr = skimage.filters.threshold_mean(img)  # type: ignore[no-untyped-call]
    vals = img[img < thr / 1]
    mmin, mmax = vals.min(), vals.max()
    x = np.linspace(mmin, mmax, num=bins)
    density = stats.gaussian_kde(vals)(x)
    if densityplot:
        plt.plot(x[: bins // 2], density[: bins // 2])
        plt.grid()
    step = x[1] - x[0]
    # # TODO: check does not fail with G550E_CFTR_DMSO_1
    peaks = signal.find_peaks(density, width=2, rel_height=0.5)
    if peaks[0].any():
        result = peaks[0][0] + peaks[1]["widths"][0]
        return float(result * step)
    # Handle the case where no peaks are found
    return float((mmax / 2 + np.median(img)) / 2)


@overload
def prob(v: float, bg: float, sd: float) -> float: ...
@overload
def prob(v: ImArray, bg: float, sd: float) -> NDArray[np.float64]: ...


def prob(v: float | ImArray, bg: float, sd: float) -> float | NDArray[np.float64]:
    """Compute pixel probability of belonging to background."""
    # Using np.sqrt(2) for normalization
    result = special.erfc((v - bg) / (np.sqrt(2) * sd))
    result = np.minimum(1, result)
    # Use typing.cast to explicitly inform mypy
    if isinstance(v, float):
        return cast(float, result)
    return cast(NDArray[np.float64], result)


def fit_gaussian(vals: NDArray[np.float64 | np.int_]) -> tuple[float, float]:
    """Estimate mean and standard deviation using a Gaussian fit.

    The function fits a Gaussian distribution to a given array of values and
    estimates the mean and standard deviation of the distribution. This process
    involves constructing a histogram of the input values, fitting the Gaussian
    model to the histogram, and optimizing the parameters of the Gaussian
    function to best match the data.

    Parameters
    ----------
    vals : NDArray[np.float64 | np.int_]
        A one-dimensional NumPy array containing the data values for which the
        Gaussian distribution parameters (mean and standard deviation) are to be
        estimated.

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
    The Gaussian fitting process involves constructing a histogram from the
    input array and then fitting a Gaussian model to this histogram. The
    optimization is performed using the least squares method to minimize the
    difference between the histogram of the data and the Gaussian function
    defined as:

        f(x) = amplitude * exp(-0.5 * ((x - mean) / sigma)^2) + offset

    where amplitude, mean, sigma, and offset are parameters of the Gaussian
    function, with 'mean' and 'sigma' being the primary parameters of interest
    in this function.

    This function relies on the `leastsq` optimization function from
    `scipy.optimize` and the method `norm.fit` from `scipy.stats.distributions`
    to estimate initial parameters for the optimization process.
    """

    def gaussian_fit_func(
        params: list[float], x: float | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        amplitude, mean, sigma, offset = params
        return amplitude * np.exp(-0.5 * ((x - mean) / sigma) ** 2) + offset

    def fit_error_func(
        params: list[float], x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> NDArray[np.float64]:
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
def calculate_bg_iteratively(
    frame: NDArray[np.float64],
    bgmax: None | np.float64 = None,
    *,
    probplot: bool = False,
) -> BgResult:
    """Refine iteratively background estimate of an image frame using Gaussian fitting.

    This function takes a single image frame, performs an initial estimate of
    the background using the median value, and then iteratively refines this
    estimate by applying a Gaussian fit on values beneath this background level.
    The process is repeated until convergence is achieved, enhancing the
    accuracy of the background estimate.

    Parameters
    ----------
    frame : NDArray[np.float64]
        The image frame for which the background estimate needs to be refined.
    bgmax : None | np.float64, optional
        Maximum value used from `frame` for background estimation. Defaults to
        None, using the mean of all pixels.
    probplot : bool, optional
        If True, generates a Q-Q plot to assess Gaussian fit. Default is False.

    Returns
    -------
    BgResult
        Comprise the values for bg, sd and iqr, and the figure object of the
        probability plot if `probplot` is True; otherwise, None.

    Examples
    --------
    >>> import numpy as np
    >>> from scipy import ndimage
    >>> frame = np.random.normal(loc=100, scale=10, size=(256, 256))
    >>> bg_result = bg_refine_iteratively(frame)
    >>> print(f"Refined Background: {bg_result.bg}, Standard Deviation: {bg_result.sd}")
    """
    prob_threshold = 0.005
    # Initial background estimate using the median of the frame
    bg_max = 1.5 * np.mean(frame) if bgmax is None else bgmax
    vals_below_bg_max = frame[frame < bg_max]
    # `fit_gaussian` is 40% faster than scipy
    # and its first guess is closer to the final result
    bg_initial, sd_ = fit_gaussian(vals_below_bg_max)
    # Iterative refinement
    bg_break = bg_initial
    with ProgressBar():  # type: ignore[no-untyped-call]
        for _i in range(100):  # Maximum of 100 iterations for refinement
            # Filtering using the current background estimate
            prob_frame = prob(frame, bg_break, sd_)
            mask = (
                ndimage.percentile_filter(prob_frame, percentile=1, size=2)
                > prob_threshold
            )
            # TODO: mask = geometric_mean_filter(prob_frame, kernel_size=5.0) > .1
            # TODO: mask = prob_frame > prob_threshold
            filtered_frame = frame[mask]
            bg_updated, sd_ = stats.distributions.norm.fit(filtered_frame, method="MM")
            if np.isclose(bg_updated, bg_break, atol=1e-6):  # Tolerance for convergence
                break
            bg_break = bg_updated
    # Return also a probability plot
    if probplot:
        fig = plt.figure(figsize=(12, 4))
        ax1 = fig.add_subplot(131)
        # Generate the Gaussian distribution
        xmin, xmax = filtered_frame.min(), filtered_frame.max()
        x = np.linspace(xmin, xmax, 100)
        p = stats.norm.pdf(x, bg_updated, sd_)
        bs, ss = stats.distributions.norm.fit(filtered_frame, method="MLE")
        print(bs, ss)
        ps = stats.norm.pdf(x, bs, ss)
        ax1.hist(filtered_frame, bins=20, density=True, alpha=0.6, color="g")
        # Plot the Gaussian fit
        ax1.plot(x, p, "r", linewidth=2)
        ax1.plot(x, ps, "r-", linewidth=1)
        # Set the title and labels
        ax1.set_title("Histogram with Gaussian Fit")
        ax1.set_xlabel("Value")
        ax1.set_ylabel("Frequency")
        ax2 = fig.add_subplot(132)
        ax2.set_title("Probplot")
        stats.probplot(filtered_frame, plot=ax2, rvalue=True)
        ax3 = fig.add_subplot(133)
        masked = (frame * mask).clip(np.min(frame))
        img0 = ax3.imshow(masked)
        plt.colorbar(img0, ax=ax3, orientation="horizontal")
        fig.tight_layout()
        figs = [fig]
        print(xmax)
    else:
        figs = None
    iqr = np.percentile(filtered_frame, [25, 50, 75])
    return BgResult(bg_updated, sd_, iqr, figs)


def geometric_mean_filter(image: ImArray, kernel_size: float) -> ImArray:
    """Apply a geometric mean filter to an image.

    It uses a neighborhood defined by kernel_size.

    Parameters
    ----------
    image : ImArray
        The input image to be filtered.
    kernel_size : float
        The diameter of the neighborhood used for the filter, which defines the
        size of the disk-shaped kernel.

    Returns
    -------
    geometric_mean_image : ImArray
        The image after applying the geometric mean filter, having the same
        shape as the input image.
    """
    # Ensure the image is in float format to avoid issues with log(0)
    image = image.astype(np.float64)
    # Avoid log(0) by replacing zero with a very small number
    image[image == 0] = np.finfo(np.float64).eps
    # Logarithm of the image
    log_image = np.log(image)
    # Create a disk-shaped kernel
    kernel = skimage.morphology.disk(kernel_size).astype(float)  # type: ignore[no-untyped-call]
    n = np.sum(kernel)  # Total weight, or number of ones in the kernel
    print(n)
    # Apply convolution with the kernel on the logged image
    log_sum_image = ndimage.convolve(log_image, kernel, mode="constant", cval=0) / n
    # Exponential to invert the logarithm
    return np.exp(log_sum_image)  # type: ignore[no-any-return]
