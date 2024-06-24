"""Main library module.

Contains functions for the analysis of multichannel timelapse images. It can
be used to apply dark, flat correction; segment cells from bg; label cells;
obtain statistics for each label; compute ratio and ratio images between
channels.
"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Any, TypeVar, cast

import matplotlib as mpl
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import ndimage, signal  # type: ignore[import-untyped]
from skimage import feature, filters, measure, morphology, segmentation, transform
from tifffile import TiffFile

from .segmentation import BgParams, bg

threshold_method_choices = ["yen", "li"]

ImArray = TypeVar("ImArray", NDArray[np.int_], NDArray[np.float64], NDArray[np.bool_])
Im = TypeVar("Im", NDArray[np.int_], NDArray[np.float64])
# MAYBE: DIm eq TypeVar("DIm", Dict[str, Im])
Kwargs = dict[str, str | int | float | bool | None]
AXES_LENGTH_4D = 4
AXES_LENGTH_3D = 3
AXES_LENGTH_2D = 2


def read_tiff(fp: Path, channels: Sequence[str]) -> tuple[dict[str, ImArray], int, int]:
    """Read multichannel TIFF timelapse image.

    Parameters
    ----------
    fp : Path
        File (TIF format) to be opened.
    channels: Sequence[str]
        List a name for each channel.

    Returns
    -------
    d_im : dict[str, ImArray]
        Dictionary of images. Each keyword represents a channel, named
        according to `channels` string list.
    n_channels : int
        Number of channels.
    n_times : int
        Number of timepoints.

    Raises
    ------
    Exception
        When number of channels and total length of TIFF sequence does not match.

    Examples
    --------
    >>> d_im, n_channels, n_times = read_tiff('tests/data/1b_c16_15.tif', \
            channels=['G', 'R', 'C'])
    >>> n_channels, n_times
    (3, 4)

    """
    n_channels = len(channels)
    with TiffFile(fp) as tif:
        im = tif.asarray()
        axes = tif.series[0].axes
    idx = axes.rfind("T")
    n_times = im.shape[idx] if idx >= 0 else 1
    if im.shape[axes.rfind("C")] % n_channels:
        msg = "n_channel mismatch total length of TIFF sequence"
        raise ValueError(msg)
    else:
        d_im = {}
        for i, ch in enumerate(channels):
            # FIXME: must be 'TCYX' or 'ZCYX'
            if len(axes) == AXES_LENGTH_4D:
                d_im[ch] = im[:, i]  # im[i::n_channels]
            elif len(axes) == AXES_LENGTH_3D:
                d_im[ch] = im[np.newaxis, i]
        print(d_im["G"].shape)
        return d_im, n_channels, n_times


def d_show(d_im: dict[str, ImArray], **kws: Any) -> Figure:  # noqa: ANN401
    """Imshow for dictionary of image (d_im). Support plt.imshow kws."""
    max_rows = 9
    n_channels = len(d_im.keys())
    first_channel = d_im[next(iter(d_im.keys()))]
    n_times = len(first_channel)
    if n_times <= max_rows:
        rng = range(n_times)
        n_rows = n_times
    else:
        step = np.ceil(n_times / max_rows).astype(int)
        rng = range(0, n_times, step)
        n_rows = len(rng)
    fig = plt.figure(figsize=(16, 16))
    for n, ch in enumerate(sorted(d_im.keys())):
        for i, r in enumerate(rng):
            ax = fig.add_subplot(n_rows, n_channels, i * n_channels + n + 1)
            img0 = ax.imshow(d_im[ch][r], **kws)
            plt.colorbar(img0, ax=ax, orientation="vertical", pad=0.02, shrink=0.85)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(f"{ch} @ t = {r}")
    plt.subplots_adjust(wspace=0.2, hspace=0.02, top=0.9, bottom=0.1, left=0, right=1)
    return fig


def d_median(d_im: dict[str, ImArray]) -> dict[str, ImArray]:
    """Median filter on dictionary of image (d_im).

    Same to skimage.morphology.disk(1) and to median filter of Fiji/ImageJ
    with radius=0.5.

    Parameters
    ----------
    d_im : dict[str, ImArray]
        dict of images

    Return
    ------
    d_im : dict[str, ImArray]
        dict of images preserve dtype of input

    Raises
    ------
    Exception
        When ImArray is neither a single image nor a stack.

    """
    d_out = {}
    for k, im in d_im.items():
        if im.ndim not in [AXES_LENGTH_2D, AXES_LENGTH_3D]:
            msg = "Only for single image or stack (3D)."
            raise ValueError(msg)
        disk = morphology.disk(1)  # type: ignore[no-untyped-call]
        if im.ndim == AXES_LENGTH_3D:
            sel = np.conj((np.zeros((3, 3)), disk, np.zeros((3, 3))))
            d_out[k] = ndimage.median_filter(im, footprint=sel)
        elif im.ndim == AXES_LENGTH_2D:
            d_out[k] = ndimage.median_filter(im, footprint=disk)
    return d_out


def d_shading(
    d_im: dict[str, ImArray],
    dark: dict[str, ImArray] | NDArray[np.float64],
    flat: dict[str, ImArray] | NDArray[np.float64],
    *,
    clip: bool = True,
) -> dict[str, ImArray]:
    """Shading correction on d_im.

    Subtract dark; then divide by flat.

    Works either with flat or d_flat
    Need also dark for each channel because it can be different when using
    different acquisition times.

    Parameters
    ----------
    d_im : dict[str, ImArray]
        Dictionary of images.
    dark : dict[str, ImArray] | NDArray[np.float64]
        Dark image (either a 2D image or 2D d_im).
    flat : dict[str, ImArray] | NDArray[np.float64]
        Flat image (either a 2D image or 2D d_im).
    clip : bool
        Boolean for clipping values >=0.

    Returns
    -------
    dict[str, ImArray]
        Corrected d_im.

    """
    # TODO inplace=True tosave memory
    # assertion type(dark) == np.ndarray or dark.keys() == d_im.keys(), raise_msg
    # assertion type(flat) == np.ndarray or flat.keys() == d_im.keys(),
    # raise_msg will be replaced by type checking.
    d_cor = {}
    for k in d_im:
        d_cor[k] = d_im[k].astype(float)
        if isinstance(dark, dict):
            d_cor[k] -= dark[k]
        else:
            d_cor[k] -= dark  # numpy.ndarray
        if isinstance(flat, dict):
            d_cor[k] /= flat[k]
        else:
            d_cor[k] /= flat  # numpy.ndarray
    if clip:
        for k in d_cor:
            d_cor[k] = d_cor[k].clip(0)
    return d_cor


def d_bg(
    d_im: dict[str, Im],
    bg_params: BgParams,
    downscale: tuple[int, int] | None = None,
    *,
    clip: bool = True,
) -> tuple[dict[str, Im], pd.DataFrame, dict[str, list[list[Figure]]]]:
    """Bg segmentation for d_im.

    Parameters
    ----------
    d_im : dict[str, Im]
        desc
    bg_params : BgParams
        An instance of BgParams containing the parameters for the segmentation.
    downscale : tuple[int, int] | None
        Tupla, x, y are downscale factors for rows, cols (default=None).
    clip : bool, optional
        Boolean (default=True) for clipping values >=0.

    Returns
    -------
    d_cor : dict[str, Im]
        Dictionary of images subtracted for the estimated bg.
    bgs : pd.DataFrame
        Median of the estimated bg; columns for channels and index for time
        points.
    figs : dict[str, list[list[Figure]]]
        List of (list ?) of figures.

    """
    d_bg = defaultdict(list)
    d_cor = defaultdict(list)
    d_fig = defaultdict(list)
    dd_cor: dict[str, Im] = {}
    for k in d_im:
        for t, im in enumerate(d_im[k]):
            im_for_bg = im
            if downscale:
                im_for_bg = transform.downscale_local_mean(im, downscale)  # type: ignore[no-untyped-call]
            bg_result = bg(im_for_bg, bg_params=bg_params)
            med = bg_result.iqr[1]
            if bg_result.figures:
                d_fig[k].append(bg_result.figures)
            d_bg[k].append(med)
            d_cor[k].append(d_im[k][t] - med)
        dd_cor[k] = np.array(d_cor[k])
    if clip:
        for k in d_cor:
            dd_cor[k] = dd_cor[k].clip(0)
    bgs = pd.DataFrame({k: np.array(v) for k, v in d_bg.items()})
    return dd_cor, bgs, d_fig


def d_mask_label(  # noqa: PLR0913
    d_im: dict[str, ImArray],
    min_size: int | None = 640,
    channels: tuple[str, ...] = ("C", "G", "R"),
    threshold_method: str = "yen",
    *,
    wiener: bool = False,
    watershed: bool = False,
    clear_border: bool = False,
    randomwalk: bool = False,
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
    d_im : dict[str, ImArray]
        desc
    min_size : int | None, optional
        Objects smaller than min_size (default=640 pixels) are discarded from mask.
    channels : tuple[str, ...], optional
        List a name for each channel.
    threshold_method : str, optional
        Threshold method applied to the geometric average plane-by-plane (default=yen).
    wiener : bool, optional
        Boolean for wiener filter (default=False).
    watershed : bool, optional
        Boolean for watershed on labels (default=False).
    clear_border :  bool, optional
        Whether to filter out objects near the 2D image edge (default=False).
    randomwalk :  bool, optional
        Use random_walker instead of watershed post-ndimage-EDT (default=False).

    Raises
    ------
    ValueError
        If threshold_method is not one of ['yen', 'li'].

    Notes
    -----
    Side effects:
        Add a 'label' key to the d_im.

    """
    if threshold_method not in threshold_method_choices:
        msg = f"threshold_method must be one of {threshold_method_choices}"
        raise ValueError(msg)

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
        threshold_function = filters.threshold_yen
    elif threshold_method == "li":
        threshold_function = filters.threshold_li  # type: ignore[assignment]
    mask = []
    for _, im in enumerate(ga_wiener):
        m = im > threshold_function(im)  # type: ignore[no-untyped-call]
        m = morphology.remove_small_objects(m, min_size=min_size)  # type: ignore[no-untyped-call]
        m = morphology.closing(m)
        # clear border always
        if clear_border:
            m = segmentation.clear_border(m)  # type: ignore[no-untyped-call]
        mask.append(m)
    d_im["mask"] = np.array(mask)
    labels, n_labels = ndimage.label(mask)
    # TODO if any timepoint mask is empty cluster labels
    d_im["labels"] = labels

    if watershed:
        process_watershed(d_im, channels, randomwalk=randomwalk)


def process_watershed(
    d_im: dict[str, ImArray],
    channels: tuple[str, ...],
    *,
    randomwalk: bool = False,
) -> None:
    """Apply watershed segmentation algorithm to a given image.

    This function takes a pre-processed image `d_im`, a sequence of channels `channels`
    to be used for the segmentation, a structuring element `ga_wiener`, a mask `mask`,
    a time step `time`, and an optional `randomwalk` function or string to specify the
    random walker method. If `labels` is not provided, it is initialized as an empty
    numpy array.

    """
    # TODO: label can change from time to time, Need more robust here. may
    # use props[0].label == 1
    # TODO: Voronoi? depends critically on max_diameter.
    distance = ndimage.distance_transform_edt(d_im["mask"][0])
    pr = measure.regionprops(  # type: ignore[no-untyped-call]
        d_im["labels"][0], intensity_image=d_im[channels[0]][0]
    )
    max_diameter = pr[0].equivalent_diameter
    size = max_diameter * 2.20
    for p in pr[1:]:
        max_diameter = max(max_diameter, p.equivalent_diameter)
    print(max_diameter)
    for time, (d, lbl) in enumerate(zip(distance, d_im["labels"], strict=True)):
        local_maxi = feature.peak_local_max(  # type: ignore[call-arg, no-untyped-call]
            d,
            labels=lbl,
            footprint=np.ones((size, size)),
            min_distance=size,
            indices=False,
            exclude_border=False,
        )
        markers = measure.label(local_maxi)  # type: ignore[no-untyped-call]
        print(np.unique(markers))
        if randomwalk:
            markers[~d_im["mask"][time]] = -1
            labels_ws = segmentation.random_walker(d_im["mask"][time], markers)
        else:
            labels_ws = morphology.watershed(-d, markers, mask=lbl)  # type: ignore[attr-defined]
    d_im["labels"][time] = labels_ws


def d_ratio(
    d_im: dict[str, ImArray],
    name: str = "r_cl",
    channels: tuple[str, str] = ("C", "R"),
    radii: tuple[int, int] = (7, 3),
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
    d_im : dict[str, ImArray]
        desc
    name : str, optional
        Name (default='r_cl') for the new key.
    channels : tuple[str, str], optional
        Names for the two channels (Numerator, Denominator) (default=('C', 'R')).
    radii : tuple[int, int], optional
        Each element contain a radius value for a median filter cycle (default=(7, 3)).

    Notes
    -----
    Add a key named "name" and containing the calculated ratio to d_im.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0/0 and num/0 can both happen.
        ratio = np.array(d_im[channels[0]] / d_im[channels[1]], dtype=(float))
    for i, r in enumerate(ratio):
        np.nan_to_num(r, copy=False, posinf=0, neginf=0)
        filtered_r = r
        for radius in radii:
            filtered_r = ndimage.median_filter(filtered_r, radius)
        ratio[i] = filtered_r * d_im["mask"][i]
    d_im[name] = ratio


def d_meas_props(  # noqa: PLR0913
    d_im: dict[str, Im],
    channels: Sequence[str] = ("C", "G", "R"),
    channels_cl: tuple[str, str] = ("C", "R"),
    channels_ph: tuple[str, str] = ("G", "C"),
    radii: tuple[int, int] | None = None,
    *,
    ratios_from_image: bool = True,
) -> tuple[dict[np.int32, pd.DataFrame], dict[str, list[list[Any]]]]:
    """Calculate pH and cl ratios and labelprops.

    Parameters
    ----------
    d_im : dict[str, Im]
        desc
    channels : Sequence[str], optional
        All d_im channels (default=('C', 'G', 'R')).
    channels_cl : tuple[str, str], optional
        Numerator and denominator channels for cl ratio (default=('C', 'R')).
    channels_ph : tuple[str, str], optional
        Numerator and denominator channels for pH ratio (default=('G', 'C')).
    radii : tuple[int, int] | None, optional
        Radii of the optional median average performed on ratio images (default=None).
    ratios_from_image : bool, optional
        Boolean for executing d_ratio i.e. compute ratio images (default=True).

    Returns
    -------
    meas : dict[np.int32, pd.DataFrame]
        For each label in labels: {'label': df}.
        DataFrame columns are: mean intensity of all channels,
        'equivalent_diameter', 'eccentricity', 'area', ratios from the mean
        intensities and optionally ratios from ratio-image.
    pr : dict[str, list[list[Any]]]
        For each channel: {'channel': [props]} i.e. {'channel': [time][label]}.

    """
    pr: dict[str, list[list[Any]]] = defaultdict(list)
    for ch in channels:
        pr[ch] = []
        for time, label_im in enumerate(d_im["labels"]):
            im = d_im[ch][time]
            props = measure.regionprops(label_im, intensity_image=im)  # type: ignore[no-untyped-call]
            pr[ch].append(props)
    meas = {}
    # labels are 3D and "0" is always label for background
    labels = np.unique(d_im["labels"])[1:]
    for lbl in labels:
        idx = []
        d = defaultdict(list)
        for time, props in enumerate(pr[channels[0]]):
            try:
                i_label = [prop.label == lbl for prop in props].index(True)
                prop_ch0 = props[i_label]
                idx.append(time)
                d["equivalent_diameter"].append(prop_ch0.equivalent_diameter)
                d["eccentricity"].append(prop_ch0.eccentricity)
                d["area"].append(prop_ch0.area)
                for ch in pr:
                    d[ch].append(pr[ch][time][i_label].mean_intensity)
            except ValueError:
                pass  # label is absent in this timepoint
        res_df = pd.DataFrame({k: np.array(v) for k, v in d.items()}, index=idx)
        res_df["r_cl"] = res_df[channels_cl[0]] / res_df[channels_cl[1]]
        res_df["r_pH"] = res_df[channels_ph[0]] / res_df[channels_ph[1]]
        meas[lbl] = res_df
    if ratios_from_image:
        kwargs = {}
        if radii:
            kwargs["radii"] = radii
        d_ratio(d_im, "r_cl", channels=channels_cl, **kwargs)
        d_ratio(d_im, "r_pH", channels=channels_ph, **kwargs)
        r_ph = []
        r_cl = []
        for time, (ph, cl) in enumerate(zip(d_im["r_pH"], d_im["r_cl"], strict=True)):
            r_ph.append(ndimage.median(ph, d_im["labels"][time], index=labels))
            r_cl.append(ndimage.median(cl, d_im["labels"][time], index=labels))
        ratios_ph = np.array(r_ph)
        ratios_cl = np.array(r_cl)
        for lbl in meas:
            res_df = pd.DataFrame(
                {
                    "r_pH_median": ratios_ph[:, lbl - 1],
                    "r_cl_median": ratios_cl[:, lbl - 1],
                }
            )
            # concat only on index that are present in both
            meas[lbl] = pd.concat([meas[lbl], res_df], axis=1, join="inner")
    return meas, pr


def d_plot_meas(
    bgs: pd.DataFrame, meas: dict[np.int32, pd.DataFrame], channels: Sequence[str]
) -> Figure:
    """Plot meas object.

    Plot r_pH, r_cl, mean intensity for each channel and estimated bg over
    timepoints for each label (color coded).

    Parameters
    ----------
    bgs : pd.DataFrame
        Estimated bg returned from d_bg()
    meas : dict[np.int32, pd.DataFrame]
        meas object returned from d_meas_props().
    channels : Sequence[str]
        All bgs and meas channels (default=['C', 'G', 'R']).

    Returns
    -------
    Figure
        Figure.

    """
    ncols = 2
    n_axes = len(channels) + 3  # 2 ratios and 1 bg axes
    nrows = int(np.ceil(n_axes / ncols))
    # colors by segmented r.o.i. id and channel names
    id_colors = mpl.cm.Set2.colors  # type: ignore[attr-defined]
    ch_colors = {
        k: k.lower() if k.lower() in mpl.colors.BASE_COLORS else "k" for k in channels
    }
    fig = plt.figure(figsize=(ncols * 5, nrows * 3))
    axes = cast(np.ndarray[Any, Any], fig.subplots(nrows, ncols))
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
        bgs.plot(ax=axes[nrows - 1, ncols - 1], grid=True, color=ch_colors)
    else:
        axes.flat[-3].set_xlabel("time")
        axes.flat[-2].set_xlabel("time")
        bgs.plot(ax=axes[nrows - 1, ncols - 2], grid=True, color=ch_colors)
        ax = list(chain(*axes))[-1]
        ax.remove()

    fig.tight_layout()
    return fig


def plt_img_profile(  # noqa: PLR0915
    img: ImArray,
    title: str | None = None,
    hpix: pd.DataFrame | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Summary graphics for Flat-Bias images.

    Parameters
    ----------
    img : ImArray
        Image of Flat or Bias.
    title : str | None, optional
        Title of the figure (default=None).
    hpix : pd.DataFrame | None, optional
        Identified hot pixels (as empty or not empty df) (default=None).
    vmin : float | None, optional
        Minimum value (default=None).
    vmax : float | None, optional
        Maximum value (default=None).

    Returns
    -------
    Figure
        Profile plot.

    """
    # definitions for the axes
    ratio = img.shape[0] / img.shape[1]
    left, width = 0.05, 0.6
    bottom, height = 0.05, 0.6 * ratio
    spacing, marginal = 0.05, 0.25
    rect_im = [left, bottom, width, height]
    rect_px = [left, bottom + height, width, marginal]
    rect_py = [left + width, bottom, marginal, height]
    rect_ht = [
        left + width + spacing,
        bottom + height + spacing,
        marginal,
        marginal / ratio,
    ]
    fig = plt.figure(figsize=(8.0, 8.0))  # * (0.4 + 0.6 * ratio)))

    if title:
        kw = {"weight": "bold", "ha": "left"}
        fig.suptitle(title, fontsize=12, x=spacing * 2, **kw)

    ax = fig.add_axes(rect_im)  # type: ignore[call-overload]
    with plt.style.context("_mpl-gallery"):
        ax_px = fig.add_axes(rect_px, sharex=ax)  # type: ignore[call-overload]
        ax_py = fig.add_axes(rect_py, sharey=ax)  # type: ignore[call-overload]
        ax_hist = fig.add_axes(rect_ht)  # type: ignore[call-overload]
    ax_cm = fig.add_axes([0.45, 0.955, 0.3, 0.034])  # type: ignore[call-overload]
    # sigfig: ax_hist.set_title("err: " + str(sigfig.
    # sigfig: round(da.std(da.from_zarr(zim)).compute(), sigfigs=3)))

    def img_hist(  # noqa: PLR0913
        im: ImArray,
        ax: Axes,
        ax_px: Axes,
        ax_py: Axes,
        axh: Axes,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> mpl.image.AxesImage:
        ax_px.tick_params(axis="x", labelbottom=False, labeltop=True, top=True)
        ax_py.tick_params(
            axis="y", right=True, labelright=True, left=False, labelleft=False
        )
        ax.tick_params(axis="y", labelleft=False, right=True)
        ax.tick_params(axis="x", top=True, labelbottom=False)
        if vmin is None or vmax is None:  # both must be provided
            vmi, vma = np.percentile(im, [18.4, 81.6])  # 1/e (66.6 %)
        else:
            vmi, vma = vmin, vmax
        img = ax.imshow(im, vmin=vmi, vmax=vma, cmap="turbo")
        ax_px.plot(im.mean(axis=0), lw=4, alpha=0.5)
        ymin = round(im.shape[0] / 2 * 0.67)
        ymax = round(im.shape[0] / 2 * 1.33)
        xmin = round(im.shape[1] / 2 * 0.67)
        xmax = round(im.shape[1] / 2 * 1.33)
        ax_px.plot(im[ymin:ymax, :].mean(axis=0), alpha=0.7, c="k")
        ax_px.xaxis.set_label_position("top")
        ax.set_xlabel("X")
        ax.axvline(xmin, c="k")
        ax.axvline(xmax, c="k")
        ax.axhline(ymin, c="k")
        ax.axhline(ymax, c="k")
        ax.yaxis.set_label_position("left")
        ax.set_ylabel("Y")
        ax_py.plot(im.mean(axis=1), range(im.shape[0]), lw=4, alpha=0.5)
        ax_py.plot(im[:, xmin:xmax].mean(axis=1), range(im.shape[0]), alpha=0.7, c="k")
        axh.hist(
            im.ravel(),
            bins=max(int(im.max() - im.min()), 25),
            log=True,
            alpha=0.6,
            lw=4,
            histtype="bar",
        )
        return img

    if hpix is not None and not hpix.empty:
        ax.plot(hpix["x"], hpix["y"], "+", mfc="gray", mew=2, ms=14)

    im2c = img_hist(img, ax, ax_px, ax_py, ax_hist, vmin, vmax)
    ax_cm.axis("off")
    fig.colorbar(
        im2c, ax=ax_cm, fraction=0.99, shrink=0.99, aspect=4, orientation="horizontal"
    )
    return fig


def plt_img_profile_2(img: ImArray, title: str | None = None) -> Figure:
    """Summary graphics for Flat-Bias images.

    Parameters
    ----------
    img : ImArray
        Image of Flat or Bias.
    title : str | None, optional
        Title of the figure  (default=None).

    Returns
    -------
    Figure
        Profile plot.

    """
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(3, 3)
    ax = fig.add_subplot(gs[0:2, 0:2])
    vmi, vma = np.percentile(img, [18.4, 81.6])  # 1/e (66.6 %)
    ax.imshow(img, vmin=vmi, vmax=vma, cmap="turbo")
    ymin = round(img.shape[0] / 2 * 0.67)
    ymax = round(img.shape[0] / 2 * 1.33)
    xmin = round(img.shape[1] / 2 * 0.67)
    xmax = round(img.shape[1] / 2 * 1.33)
    ax.axvline(xmin, c="k")
    ax.axvline(xmax, c="k")
    ax.axhline(ymin, c="k")
    ax.axhline(ymax, c="k")
    ax1 = fig.add_subplot(gs[2, 0:2])
    ax1.plot(img.mean(axis=0))
    ax1.plot(img[ymin:ymax, :].mean(axis=0), alpha=0.2, lw=2, c="k")
    ax2 = fig.add_subplot(gs[0:2, 2])
    ax2.plot(
        img[:, xmin:xmax].mean(axis=1), range(img.shape[0]), alpha=0.2, lw=2, c="k"
    )
    ax2.plot(img.mean(axis=1), range(img.shape[0]))
    axh = fig.add_subplot(gs[2, 2])
    axh.hist(img.ravel(), bins=max(int(img.max() - img.min()), 25), log=True)
    if title:
        kw = {"weight": "bold", "ha": "left"}
        fig.suptitle(title, fontsize=12, **kw)
    return fig


def hotpixels(bias: ImArray, n_sd: int = 20) -> pd.DataFrame:
    """Identify hot pixels in a bias-dark frame.

    After identification of first outliers recompute masked average and std
    until convergence.

    Parameters
    ----------
    bias : ImArray
        Usually the median over a stack of 100 frames.
    n_sd : int
        Number of SD above mean (masked out of hot pixels) value.

    Returns
    -------
    pd.DataFrame
        y, x positions and values of hot pixels.

    """
    ave = bias.mean()
    std = bias.std()
    m = bias > (ave + n_sd * std)
    n_hpix = m.sum()
    while True:
        m_ave = np.ma.masked_array(bias, m).mean()  # type: ignore[no-untyped-call]
        m_std = np.ma.masked_array(bias, m).std()  # type: ignore[no-untyped-call]
        m = bias > m_ave + n_sd * m_std
        if n_hpix == m.sum():
            break
        n_hpix = m.sum()
    w = np.where(m)
    hpix_df = pd.DataFrame({"y": w[0], "x": w[1]})
    hpix_df = hpix_df.assign(val=lambda row: bias[row.y, row.x])
    return hpix_df


def correct_hotpixel(
    img: ImArray, y: int | NDArray[np.int_], x: int | NDArray[np.int_]
) -> None:
    """Correct hot pixels in a frame.

    Substitute indicated position y, x with the median value of the 4 neighbor
    pixels.

    Parameters
    ----------
    img : ImArray
        Frame (2D) image.
    y : int | NDArray[np.int_]
        y-coordinate(s).
    x : int | NDArray[np.int_]
        x-coordinate(s).

    """
    if img.ndim == AXES_LENGTH_2D:
        v1 = img[y - 1, x]
        v2 = img[y + 1, x]
        v3 = img[y, x - 1]
        v4 = img[y, x + 1]
        correct = np.median([v1, v2, v3, v4])
        img[y, x] = correct
