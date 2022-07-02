"""Main library module.

Contains functions for the analysis of multichannel timelapse images. It can
be used to apply dark, flat correction; segment cells from bg; label cells;
obtain statistics for each label; compute ratio and ratio images between
channels.
"""
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import TypeVar
from typing import Union

import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage  # type: ignore
import skimage.feature  # type: ignore
import skimage.segmentation  # type: ignore
import skimage.transform  # type: ignore
import tifffile  # type: ignore
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore
from scipy import signal
from skimage import filters
from skimage.morphology import disk  # type: ignore


ImArray = TypeVar("ImArray", NDArray[np.float_], NDArray[np.int_])

# TODO: https://towardsdatascience.com/creating-custom-plotting-functions-with-matplotlib-1f4b8eba6aa1


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


def plot_im_series(
    im: ImArray,
    cmap: matplotlib.colors.Colormap = matplotlib.colors.BASE_COLORS,
    horizontal: bool = True,
    **kw: Any,
) -> None:
    """Plot a image series with a maximum of 9 elements.

    ..note:: Consider deprecation. Use d_show() instead.

    """
    if horizontal:
        fig = plt.figure(figsize=(12, 5.6))
        s = 100 + len(im) * 10 + 1
    else:
        fig = plt.figure(figsize=(5.6, 12))
        s = len(im) * 100 + 10 + 1
    for i, img in enumerate(im):
        ax = fig.add_subplot(s + i)
        # Switched from plt.cmd to ax.cmd
        ax.imshow(img, cmap=cmap, **kw)
        plt.axis("off")  # type: ignore
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)


def plot_otsu(
    im: ImArray, cmap: Optional[mpl.colors.Colormap] = None
) -> NDArray[np.bool_]:
    """Otsu threshold and plot im_series.

    .. note:: Consider deprecation.

    """
    if not cmap:
        cmap = plt.cm.gray  # type: ignore
    val = filters.threshold_otsu(im)
    mask = im > val
    plot_im_series(im * mask, cmap=cmap)
    return np.array(mask)


def zproject(im: ImArray, func: Callable[[Any], Any] = np.median) -> ImArray:
    """Perform z-projection of a 3D image.

    func must support axis= and out= API like np.median, np.mean, np.percentile

    Parameters
    ----------
    im : ImArray
        Image (pln, row, col).

    func
        Function (default: np.median).

    Returns
    -------
    np.array(row, col)
        2D (projected) image (median by default). Preserve dtype of input.

    Raises
    ------
    ValueError
        If the input image is not 3D.

    """
    if im.ndim != 3 or len(im) != im.shape[0]:
        raise ValueError("Input must be 3D-grayscale (pln, row, col)")
    # maintain same dtype as input im; odd and even
    zproj = np.zeros(im.shape[1:]).astype(im.dtype)
    func(im[1:], axis=0, out=zproj)  # type: ignore
    return zproj


def read_tiff(fp: Path, channels: Sequence[str]) -> Tuple[Dict[str, ImArray], int, int]:
    """Read multichannel tif timelapse image.

    Parameters
    ----------
    fp : Path
        File (TIF format) to be opened.
    channels: list of string
        List a name for each channel.

    Returns
    -------
    d_im : dict
        Dictionary of images. Each keyword represents a channel, named
        according to channels string list.
    n_channels : int
        Number of channels.
    n_times : int
        Number of timepoints.

    Examples
    --------
    >>> d_im, n_channels, n_times = read_tiff('tests/data/1b_c16_15.tif', \
            channels=['G', 'R', 'C'])
    >>> n_channels, n_times
    (3, 4)

    """
    n_channels = len(channels)
    with tifffile.TiffFile(fp) as tif:
        im = tif.asarray()
        axes = tif.series[0].axes
    idx = axes.rfind("T")
    if idx >= 0:
        n_times = im.shape[idx]
    else:
        n_times = 1
    if im.shape[axes.rfind("C")] % n_channels:
        raise Exception("n_channel mismatch total lenght of tif sequence")
    else:
        d_im = {}
        for i, ch in enumerate(channels):
            # FIXME: must be 'TCYX' or 'ZCYX'
            if len(axes) == 4:
                d_im[ch] = im[:, i]  # im[i::n_channels]
            elif len(axes) == 3:
                d_im[ch] = im[np.newaxis, i]
        print(d_im["G"].shape)
        return d_im, n_channels, n_times


def d_show(d_im: Dict[str, ImArray], **kws: Any) -> plt.Figure:
    """Imshow for dictionary of image (d_im). Support plt.imshow kws."""
    max_rows = 9
    n_channels = len(d_im.keys())
    first_channel = d_im[list(d_im.keys())[0]]
    n_times = len(first_channel)
    if n_times <= max_rows:
        rng = range(n_times)
        n_rows = n_times
    else:
        step = np.ceil(n_times / max_rows).astype(int)
        rng = range(0, n_times, step)
        n_rows = len(rng)

    f = plt.figure(figsize=(16, 16))
    for n, ch in enumerate(sorted(d_im.keys())):
        for i, r in enumerate(rng):
            ax = f.add_subplot(n_rows, n_channels, i * n_channels + n + 1)
            img0 = ax.imshow(d_im[ch][r], **kws)
            plt.colorbar(  # type: ignore
                img0, ax=ax, orientation="vertical", pad=0.02, shrink=0.85
            )
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(ch + " @ t = " + str(r))
    plt.subplots_adjust(wspace=0.2, hspace=0.02, top=0.9, bottom=0.1, left=0, right=1)
    return f


def d_median(d_im: Dict[str, ImArray]) -> Dict[str, ImArray]:
    """Median filter on dictionary of image (d_im).

    Same to skimage.morphology.disk(1) and to median filter of Fiji/ImageJ
    with radius=0.5.

    Parameters
    ----------
    d_im : dict of images

    Return
    ------
    d_im : dict of images
        preserve dtype of input

    """
    d_out = {}
    for k, im in d_im.items():
        disk = skimage.morphology.disk(1)
        if im.ndim == 3:
            sel = np.conj((np.zeros((3, 3)), disk, np.zeros((3, 3))))
            d_out[k] = ndimage.median_filter(im, footprint=sel)
        elif im.ndim == 2:
            d_out[k] = ndimage.median_filter(im, footprint=disk)
        else:
            raise Exception("Only for single image or stack (3D).")
    return d_out


def d_shading(
    d_im: Dict[str, ImArray],
    dark: Union[Dict[str, ImArray], NDArray[np.float_]],
    flat: Union[Dict[str, ImArray], NDArray[np.float_]],
    clip: bool = True,
) -> Dict[str, ImArray]:
    """Shading correction on d_im.

    Subtract dark; then divide by flat.

    Works either with flat or d_flat
    Need also dark for each channel because it can be different when using
    different acquisiton times.

    Parameters
    ----------
    d_im
        Dictionary of images.
    dark : 2D image or (2D) d_im
        Dark image.
    flat : 2D image or (2D) d_im
        Flat image.
    clip : bool
        Boolean for clipping values >=0.

    Returns
    -------
    d_im
        Corrected d_im.

    """
    # TODO inplace=True tosave memory
    # assertion type(dark) == np.ndarray or dark.keys() == d_im.keys(), raise_msg
    # assertion type(flat) == np.ndarray or flat.keys() == d_im.keys(),
    # raise_msg will be replaced by type checking.
    d_cor = {}
    for k in d_im.keys():
        d_cor[k] = d_im[k].astype(float)
        if type(dark) == dict:
            d_cor[k] -= dark[k]
        else:
            d_cor[k] -= dark  # numpy.ndarray
        if type(flat) == dict:
            d_cor[k] /= flat[k]
        else:
            d_cor[k] /= flat  # numpy.ndarray
    if clip:
        for k in d_cor.keys():
            d_cor[k] = d_cor[k].clip(0)
    return d_cor


def bg(
    im: NDArray[Any],
    kind: str = "arcsinh",
    perc: float = 10.0,
    radius: Optional[int] = 10,
    adaptive_radius: Optional[int] = None,
    arcsinh_perc: Optional[int] = 80,
) -> Tuple[float, ImArray, List[Any]]:
    """Bg segmentation.

    Return median, whole vector, figures (in a [list])

    Parameters
    ----------
    im
        An image stack.
    kind : str
        Method {'arcsinh', 'entropy', 'adaptive', 'li_adaptive', 'li_li'} used for the segmentation.
    perc : float
        Perc % of max-min (default=10) for thresholding *entropy* and *arcsinh*
        methods.
    radius : int, optional
        Radius (default=10) used in *entropy* and *arcsinh* (percentile_filter)
        methods.
    adaptive_radius : int, optional
        Size for the adaptive filter of skimage (default is im.shape[1]/2).
    arcsinh_perc : int, optional
        Perc (default=80) used in the percentile_filter (scipy) whithin
        *arcsinh* method.

    Returns
    -------
    median : float
        Median of the bg masked pixels.
    pixel_values : list ?
        Values of all bg masked pixels.
    figs : {[f1], [f1, f2]}
        List of fig(s). Only entropy and arcsinh methods have 2 elements.

    """
    if adaptive_radius is None:
        adaptive_radius = int(im.shape[1] / 2)
        if adaptive_radius % 2 == 0:  # sk >0.12.0 check for even value
            adaptive_radius += 1
    if (perc < 0.0) or (perc > 100.0):
        raise Exception("perc must be in [0, 100] range")
    else:
        perc /= 100
    if kind == "arcsinh":
        lim = np.arcsinh(im)
        lim = ndimage.percentile_filter(lim, arcsinh_perc, size=radius)
        lim_ = True
        title: Any = radius, perc
        thr = (1 - perc) * lim.min() + perc * lim.max()
        m = lim < thr
    elif kind == "entropy":
        if im.dtype == float:
            lim = filters.rank.entropy(im / im.max(), disk(radius))
        else:
            lim = filters.rank.entropy(im, disk(radius))
        lim_ = True
        title = radius, perc
        thr = (1 - perc) * lim.min() + perc * lim.max()
        m = lim < thr
    elif kind == "adaptive":
        lim_ = False
        title = adaptive_radius
        f = im > filters.threshold_local(im, adaptive_radius)
        m = ~f
    elif kind == "li_adaptive":
        lim_ = False
        title = adaptive_radius
        li = filters.threshold_li(im.copy())
        m = im < li
        # # FIXME: in case m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        f = imm > filters.threshold_local(imm, adaptive_radius)
        m = ~f * m
    elif kind == "li_li":
        lim_ = False
        title = None
        li = filters.threshold_li(im.copy())
        m = im < li
        # # FIXME: in case m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        # To avoid zeros generated after first thesholding, clipping to the
        # min value of original image is needed before second thesholding.
        thr2 = filters.threshold_li(imm.clip(np.min(im)))
        m = im < thr2
        # # FIXME: in case mm = skimage.morphology.binary_closing(mm)
    pixel_values = im[m]
    iqr = np.percentile(pixel_values, [25, 50, 75])

    def plot() -> plt.Figure:
        f = plt.figure(figsize=(9, 5))
        ax1 = f.add_subplot(121)
        masked = im * m
        cmap = plt.cm.inferno  # type: ignore
        img0 = ax1.imshow(masked, cmap=cmap)
        plt.colorbar(img0, ax=ax1, orientation="horizontal")  # type:ignore
        plt.title(kind + " " + str(title) + "\n" + str(iqr))
        f.add_subplot(122)
        myhist(im[m], log=True)
        f.tight_layout()
        return f

    f1 = plot()
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
        return iqr[1], pixel_values, [f1, f2]
    else:
        return iqr[1], pixel_values, [f1]


# TODO: add new bg/fg segmentation based on conditional probability but
# working with dask arrays.


def d_bg(
    d_im: Dict[str, ImArray],
    downscale: Optional[Tuple[int, int]] = None,
    kind: str = "li_adaptive",
    clip: bool = True,
    **kw: Dict[str, Any],
) -> Tuple[
    Dict[str, ImArray],
    pd.DataFrame,
    Dict[str, List[List[plt.Figure]]],
    Dict[str, List[Any]],
]:
    """Bg segmentation for d_im.

    Parameters
    ----------
    d_im : d_im
        desc
    downscale : {None, tupla}
        Tupla, x, y are downscale factors for rows, cols.
    kind : str
        Bg method among {'li_adaptive', 'arcsinh', 'entropy', 'adaptive', 'li_li'}.
    clip : bool
        Boolean (default=True) for clipping values >=0.
    kw : dict
        Keywords passed to bg() function.

    Returns
    -------
    d_cor : d_im
        Dictionary of images subtracted for the estimated bg.
    bgs : pd.DataFrame
        Median of the estimated bg; columns for channels and index for time
        points.
    figs : list
        List of (list ?) of figures.
    d_bg_values : dict
        Background values keys are channels containing a list (for each time
        point) of list of values.

    """
    d_bg = defaultdict(list)
    d_bg_values = defaultdict(list)
    d_cor = defaultdict(list)
    d_fig = defaultdict(list)
    dd_cor: Dict[str, NDArray[Any]] = {}
    for k in d_im.keys():
        for t, im in enumerate(d_im[k]):
            if downscale:
                im = skimage.transform.downscale_local_mean(im, downscale)
            med, v, ff = bg(im, kind=kind, perc=10)
            d_bg[k].append(med)
            d_bg_values[k].append(v)
            d_cor[k].append(d_im[k][t] - med)
            d_fig[k].append(ff)
        dd_cor[k] = np.array(d_cor[k])
    if clip:
        for k in d_cor.keys():
            dd_cor[k] = dd_cor[k].clip(0)
    bgs = pd.DataFrame({k: np.array(v) for k, v in d_bg.items()})
    return dd_cor, bgs, d_fig, d_bg_values


def d_mask_label(
    d_im: Dict[str, ImArray],
    min_size: Optional[int] = 640,
    channels: Sequence[str] = ("C", "G", "R"),
    threshold_method: Optional[str] = "yen",
    wiener: Optional[bool] = False,
    watershed: Optional[bool] = False,
    clear_border: Optional[bool] = False,
    randomwalk: Optional[bool] = False,
) -> None:
    """Label cells in d_im. Add two keys, mask and label.

    Perform plane-by-plane (2D image):

    - geometric average of all channels;
    - optional wiener filter (3,3);
    - mask using threshold_method;
    - remove objects smaller than **min_size**;
    - binary closing;
    - optionally remove any object on borders;
    - label each ROI;
    - optionally perform watershed on labels.

    Parameters
    ----------
    d_im : d_im
        desc
    min_size : type, optional
        Objects smaller than min_size (default=640 pixels) are discarded from
        mask.
    channels : list of string
        List a name for each channel.
    threshold_method : {'yen', 'li'}
        Method for thresholding (skimage) the geometric average plane-by-plane.
    wiener : bool, optional
        Boolean (default=False) for wiener filter.
    watershed : bool, optional
        Boolean (default=False) for watershed on labels.
    clear_border :  bool, optional
        Boolean (default=False) for removing objects that are touching the
        image (2D) border.
    randomwalk :  bool, optional
        Boolean (default=False) for using random_walker in place of watershed
        (skimage) algorithm after ndimage.distance_transform_edt() calculation.

    Notes
    -----
    Side effects:
        Add a 'label' key to the d_im.

    """
    ga = d_im[channels[0]].copy()
    for ch in channels[1:]:
        ga *= d_im[ch]
    ga = np.power(ga, 1 / len(channels))
    if wiener:
        ga_wiener = np.zeros_like(d_im["G"])
        shape = (3, 3)  # for 3D (1, 4, 4)
        for i, im in enumerate(ga):
            ga_wiener[i] = signal.wiener(im, shape)
    else:
        ga_wiener = ga
    if threshold_method == "yen":
        threshold_function = skimage.filters.threshold_yen
    elif threshold_method == "li":
        threshold_function = skimage.filters.threshold_li
    mask = []
    for _, im in enumerate(ga_wiener):
        m = im > threshold_function(im)
        m = skimage.morphology.remove_small_objects(m, min_size=min_size)
        m = skimage.morphology.closing(m)
        # clear border always
        if clear_border:
            m = skimage.segmentation.clear_border(m)
        mask.append(m)
    d_im["mask"] = np.array(mask)
    labels, n_labels = ndimage.label(mask)
    # TODO if any timepoint mask is empty cluster labels

    if watershed:
        # TODO: label can change from time to time, Need more robust here. may
        # use props[0].label == 1
        # TODO: Voronoi? depends critically on max_diameter.
        distance = ndimage.distance_transform_edt(mask)
        pr = skimage.measure.regionprops(
            labels[0], intensity_image=d_im[channels[0]][0]
        )
        max_diameter = pr[0].equivalent_diameter
        size = max_diameter * 2.20
        for p in pr[1:]:
            max_diameter = max(max_diameter, p.equivalent_diameter)
        print(max_diameter)
        # for time, (d, l) in enumerate(zip(ga_wiener, labels)):
        for time, (d, l) in enumerate(zip(distance, labels)):
            local_maxi = skimage.feature.peak_local_max(
                d,
                labels=l,
                footprint=np.ones((size, size)),
                min_distance=size,
                indices=False,
                exclude_border=False,
            )
            markers = skimage.measure.label(local_maxi)
            print(np.unique(markers))
            if randomwalk:
                markers[~mask[time]] = -1
                labels_ws = skimage.segmentation.random_walker(mask[time], markers)
            else:
                labels_ws = skimage.morphology.watershed(-d, markers, mask=l)
            labels[time] = labels_ws
    d_im["labels"] = labels


def d_ratio(
    d_im: Dict[str, NDArray[Any]],
    name: str = "r_cl",
    channels: Tuple[str, str] = ("C", "R"),
    radii: Tuple[int, int] = (7, 3),
) -> None:
    """Ratio image between 2 channels in d_im.

    Add masked (bg=0; fg=ratio) median-filtered ratio for 2 channels. So, d_im
    must (already) contain keys for mask and the two channels.

    After ratio computation any -inf, nan and inf values are replaced with 0.
    These values should be generated (upon ratio) only in the bg. You can
    check:
    r_cl[d_im['labels']==4].min()

    Parameters
    ----------
    d_im : d_im
        desc
    name : str
        Name (default='r_cl') for the new key.
    channels : list of string
        Names (default=['C', 'R']) for the two channels [Numerator,
        Denominator].
    radii : tupla of int, optional
        Each element contain a radius value for a median filter cycle.

    Notes
    -----
    Add a key named "name" and containing the calculated ratio to d_im.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0/0 and num/0 can both happen.
        ratio = d_im[channels[0]] / d_im[channels[1]]
    for i, r in enumerate(ratio):
        np.nan_to_num(r, copy=False, posinf=0, neginf=0)
        for radius in radii:
            r = ndimage.median_filter(r, radius)
        ratio[i] = r * d_im["mask"][i]
    d_im[name] = ratio


def d_meas_props(
    d_im: Dict[str, ImArray],
    channels: Sequence[str] = ("C", "G", "R"),
    channels_cl: Tuple[str, str] = ("C", "R"),
    channels_ph: Tuple[str, str] = ("G", "C"),
    ratios_from_image: Optional[bool] = True,
    radii: Optional[Tuple[int, int]] = None,
) -> Tuple[Dict[np.int32, pd.DataFrame], Dict[str, List[Any]]]:
    """Calculate pH and cl ratios and labelprops.

    Parameters
    ----------
    d_im : d_im
        desc
    channels : list of string
        All d_im channels (default=['C', 'G', 'R']).
    channels_cl : tuple of string
        Names (default=('C', 'R')) of the numerator and denominator channels for cl ratio.
    channels_ph : tuple of string
        Names (default=('G', 'C')) of the numerator and denominator channels for pH ratio.
    ratios_from_image : bool, optional
        Boolean (default=True) for executing d_ratio i.e. compute ratio images.
    radii : (int, int), Optional
        Radii of the optional median average performed on ratio images.

    Returns
    -------
    meas : dict of pd.DataFrame
        For each label in labels: {'label': df}.
        DataFrame columns are: mean intensity of all channels,
        'equivalent_diameter', 'eccentricity', 'area', ratios from the mean
        intensities and optionally ratios from ratio-image.
    pr : dict of list of list
        For each channel: {'channel': [props]} i.e. {'channel': [time][label]}.

    """
    pr: Dict[str, List[Any]] = defaultdict(list)
    for ch in channels:
        pr[ch] = []
        for time, label_im in enumerate(d_im["labels"]):
            im = d_im[ch][time]
            props = skimage.measure.regionprops(label_im, intensity_image=im)
            pr[ch].append(props)
    meas = {}
    # labels are 3D and "0" is always label for background
    labels = np.unique(d_im["labels"])[1:]
    for label in labels:
        idx = []
        d = defaultdict(list)
        for time, props in enumerate(pr[channels[0]]):
            try:
                i_label = [prop.label == label for prop in props].index(True)
                prop_ch0 = props[i_label]
                idx.append(time)
                d["equivalent_diameter"].append(prop_ch0.equivalent_diameter)
                d["eccentricity"].append(prop_ch0.eccentricity)
                d["area"].append(prop_ch0.area)
                for ch in pr:
                    d[ch].append(pr[ch][time][i_label].mean_intensity)
            except ValueError:
                pass  # label is absent in this timepoint
        df = pd.DataFrame({k: np.array(v) for k, v in d.items()}, index=idx)
        df["r_cl"] = df[channels_cl[0]] / df[channels_cl[1]]
        df["r_pH"] = df[channels_ph[0]] / df[channels_ph[1]]
        meas[label] = df
    if ratios_from_image:
        kwargs = {}
        if radii:
            kwargs["radii"] = radii
        d_ratio(d_im, "r_cl", channels=channels_cl, **kwargs)
        d_ratio(d_im, "r_pH", channels=channels_ph, **kwargs)
        r_ph = []
        r_cl = []
        for time, (ph, cl) in enumerate(zip(d_im["r_pH"], d_im["r_cl"])):
            r_ph.append(ndimage.median(ph, d_im["labels"][time], index=labels))
            r_cl.append(ndimage.median(cl, d_im["labels"][time], index=labels))
        ratios_ph = np.array(r_ph)
        ratios_cl = np.array(r_cl)
        for label in meas:
            df = pd.DataFrame(
                {
                    "r_pH_median": ratios_ph[:, label - 1],
                    "r_cl_median": ratios_cl[:, label - 1],
                }
            )
            # concat only on index that are present in both
            meas[label] = pd.concat([meas[label], df], axis=1, join="inner")  # type: ignore
    return meas, pr


def d_plot_meas(
    bgs: pd.DataFrame, meas: Dict[np.int32, pd.DataFrame], channels: Sequence[str]
) -> plt.Figure:
    """Plot meas object.

    Plot r_pH, r_cl, mean intensity for each channel and estimated bg over
    timepoints for each label (color coded).

    Parameters
    ----------
    bgs : pd.DataFrame
        Estimated bg returned from d_bg()
    meas : dict of pd.DataFrame
        meas object returned from d_meas_props().
    channels : list of string
        All bgs and meas channels (default=['C', 'G', 'R']).

    Returns
    -------
    fig : plt.Figure
        Figure.

    """
    ncols = 2
    n_axes = len(channels) + 3  # 2 ratios and 1 bg axes
    nrows = int(np.ceil(n_axes / ncols))
    # colors by segmented r.o.i. id and channel names
    id_colors = mpl.cm.Set2.colors  # type: ignore
    ch_colors = {
        k: k.lower() if k.lower() in mpl.colors.BASE_COLORS else "k" for k in channels
    }
    fig = plt.figure(figsize=(ncols * 5, nrows * 3))
    axes = fig.subplots(nrows, ncols)  # type: ignore
    for k, df in meas.items():
        c = id_colors[(int(k) - 1) % len(id_colors)]
        axes[0, 0].plot(df["r_pH"], marker="o", color=c, label=k)
        axes[0, 1].plot(df["r_cl"], marker="o", color=c)
        if "r_pH_median" in df:
            axes[0, 0].plot(df["r_pH_median"], color=c, linestyle="--", lw=2, label="")
        if "r_cl_median" in df:
            axes[0, 1].plot(df["r_cl_median"], color=c, linestyle="--", lw=2, label="")
    axes[0, 1].set_ylabel("r_Cl")
    axes[0, 0].set_ylabel("r_pH")
    axes[0, 0].set_title("pH")
    axes[0, 1].set_title("Cl")
    axes[0, 0].grid()
    axes[0, 1].grid()
    axes[0, 0].legend()

    for n, ch in enumerate(channels, 2):
        i = n // ncols
        j = n % ncols  # * 2
        for df in meas.values():
            axes[i, j].plot(df[ch], marker="o", color=ch_colors[ch])
        axes[i, j].set_title(ch)
        axes[i, j].grid()
    if n_axes == nrows * ncols:
        axes.flat[-2].set_xlabel("time")
        axes.flat[-1].set_xlabel("time")
        bgs.plot(ax=axes[nrows - 1, ncols - 1], grid=True, color=ch_colors)  # type: ignore
    else:
        axes.flat[-3].set_xlabel("time")
        axes.flat[-2].set_xlabel("time")
        bgs.plot(ax=axes[nrows - 1, ncols - 2], grid=True, color=ch_colors)  # type: ignore
        ax = list(chain(*axes))[-1]
        ax.remove()

    fig.tight_layout()
    return fig
