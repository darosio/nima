"""Main library module.

Contains functions for the analysis of multichannel timelapse images. It can be
used to apply dark, flat correction; segment cells from bg; label cells; obtain
statistics for each label; compute ratio and ratio images between channels.
"""

from collections import defaultdict
from collections.abc import Sequence
from itertools import chain
from pathlib import Path
from typing import Any, cast

import dask.array as da
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from dask.array import Array
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from scipy import ndimage, signal  # type: ignore[import-untyped]
from skimage import (
    feature,
    filters,
    measure as skmeasure,
    morphology,
    segmentation,
    transform,
)

from . import io
from .io import Metadata
from .nima_types import DIm, ImFrame, ImSequence
from .segmentation import BgParams, calculate_bg

Kwargs = dict[str, str | int | float | bool | None]
threshold_method_choices = ["yen", "li"]
AXES_LENGTH_4D = 4
AXES_LENGTH_3D = 3
AXES_LENGTH_2D = 2


def _compute_d_im(d_im: DIm) -> DIm:
    """Compute values in d_im if they are lazy (dask)."""
    for k, v in d_im.items():
        if hasattr(v, "compute"):
            d_im[k] = v.compute()
    return d_im


def d_show(d_im: DIm, **kws: Any) -> Figure:  # noqa: ANN401
    """Imshow for dictionary of image (d_im). Support plt.imshow kws."""
    d_im = _compute_d_im(d_im)
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


def median(im: xr.DataArray) -> xr.DataArray:
    """Median filter on xarray.DataArray.

    Same to skimage.morphology.disk(1) and to median filter of Fiji/ImageJ
    with radius=0.5.

    Parameters
    ----------
    im : xr.DataArray
        Input image data array (usually TCZYX).

    Returns
    -------
    xr.DataArray
        Filtered data array preserving dtype of input.

    """
    disk_footprint = morphology.disk(1)  # type: ignore[no-untyped-call]

    def apply_median(img: NDArray[Any]) -> NDArray[Any]:
        """Apply median filter to 2D image."""
        return cast(
            "NDArray[Any]", ndimage.median_filter(img, footprint=disk_footprint)
        )

    return cast(
        "xr.DataArray",
        xr.apply_ufunc(
            apply_median,
            im,
            input_core_dims=[["Y", "X"]],
            output_core_dims=[["Y", "X"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[im.dtype],
        ),
    )


def shading(
    im: xr.DataArray,
    dark: xr.DataArray | Any,  # noqa: ANN401
    flat: xr.DataArray | Any,  # noqa: ANN401
    *,
    clip: bool = True,
) -> xr.DataArray:
    """Shading correction on xarray.DataArray.

    Subtract dark; then divide by flat.

    Parameters
    ----------
    im : xr.DataArray
        Input image data array.
    dark : xr.DataArray | Any
        Dark image (DataArray or broadcastable array/scalar).
    flat : xr.DataArray | Any
        Flat image (DataArray or broadcastable array/scalar).
    clip : bool
        Boolean for clipping values >=0.

    Returns
    -------
    xr.DataArray
        Corrected data array.

    """
    # Cast to float
    d_cor = im.astype(float)

    # Handle broadcasting for single-frame dark/flat (e.g. T=1 or Z=1)
    if isinstance(dark, xr.DataArray):
        for dim in dark.dims:
            if dark.sizes[dim] == 1 and dim in im.dims and im.sizes[dim] > 1:
                dark = dark.squeeze(dim)
    if isinstance(flat, xr.DataArray):
        for dim in flat.dims:
            if flat.sizes[dim] == 1 and dim in im.dims and im.sizes[dim] > 1:
                flat = flat.squeeze(dim)
    # Subtract dark and divide by flat
    d_cor = (d_cor - dark) / flat
    # Clip if requested
    if clip:
        d_cor = d_cor.clip(min=0)
    return d_cor


def bg(
    im: xr.DataArray,
    bg_params: BgParams,
    downscale: tuple[int, int] | None = None,
    *,
    clip: bool = True,
) -> tuple[xr.DataArray, pd.DataFrame, dict[str, list[list[Figure]]]]:
    """Bg segmentation for xarray.DataArray.

    Parameters
    ----------
    im : xr.DataArray
        Input image data array with dimensions including C (channels) and T (time).
    bg_params : BgParams
        An instance of BgParams containing the parameters for the segmentation.
    downscale : tuple[int, int] | None
        Tupla, x, y are downscale factors for rows, cols (default=None).
    clip : bool, optional
        Boolean (default=True) for clipping values >=0.

    Returns
    -------
    d_cor : xr.DataArray
        DataArray subtracted for the estimated bg.
    bgs : pd.DataFrame
        Median of the estimated bg; columns for channels and index for time
        points.
    figs : dict[str, list[list[Figure]]]
        List of (list ?) of figures.

    """
    bgs_data = defaultdict(list)
    figs_data = defaultdict(list)

    channels = im.coords["C"].to_numpy()
    times = im.coords["T"].to_numpy()

    for ch in channels:
        ch_bgs = []
        for t in range(len(times)):
            frame_da = im.sel(C=ch).isel(T=t)
            if "Z" in im.dims and im.sizes["Z"] == 1:
                frame_da = frame_da.squeeze("Z")

            frame_np = frame_da.compute().to_numpy()

            if downscale:
                frame_for_bg = transform.downscale_local_mean(frame_np, downscale)  # type: ignore[no-untyped-call]
            else:
                frame_for_bg = frame_np

            bg_result = calculate_bg(frame_for_bg, bg_params)
            med = bg_result.iqr[1]
            ch_bgs.append(med)

            if bg_result.figures:
                figs_data[str(ch)].append(bg_result.figures)

        bgs_data[str(ch)] = ch_bgs

    bgs_df = pd.DataFrame(bgs_data, index=range(len(times)))
    bg_da = xr.DataArray(
        bgs_df.values, dims=("T", "C"), coords={"T": times, "C": channels}
    )

    im_subtracted = im - bg_da

    if clip:
        im_subtracted = im_subtracted.clip(min=0)

    return im_subtracted, bgs_df, dict(figs_data)


def _wiener_2d(im: NDArray[Any]) -> NDArray[Any]:
    """Apply 2D Wiener filter."""
    return signal.wiener(im, (3, 3))  # type: ignore[no-any-return]


def _threshold_2d(im: NDArray[Any], method: str) -> NDArray[np.bool_]:
    """Apply threshold to 2D plane."""
    if method == "li":
        threshold = filters.threshold_li(im)  # type: ignore[no-untyped-call]
    else:
        threshold = filters.threshold_yen(im)  # type: ignore[no-untyped-call]
    return im > threshold  # type: ignore[no-any-return]


def _remove_small_2d(m: NDArray[np.bool_], min_size: int) -> NDArray[np.bool_]:
    """Remove small objects from binary mask."""
    return morphology.remove_small_objects(m, max_size=min_size - 1)  # type: ignore[no-any-return]


def _closing_2d(m: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Apply binary closing morphology."""
    return morphology.closing(m)  # type: ignore[no-any-return]


def _clear_border_2d(m: NDArray[np.bool_]) -> NDArray[np.bool_]:
    """Clear objects touching the border."""
    return segmentation.clear_border(m)  # type: ignore[no-untyped-call, no-any-return]


def _label_2d(m: NDArray[np.bool_]) -> NDArray[np.int32]:
    """Label connected components in binary mask."""
    labeled, _ = ndimage.label(m)
    return labeled.astype(np.int32)  # type: ignore[no-any-return]


def segment(  # noqa: PLR0913
    im: xr.DataArray,
    min_size: int | None = 640,
    channels: tuple[str, ...] = ("C", "G", "R"),
    threshold_method: str = "yen",
    *,
    wiener: bool = False,
    watershed: bool = False,
    clear_border: bool = False,
    randomwalk: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Segment cells in xarray DataArray. Returns mask and labels.

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
    im : xr.DataArray
        Input image with dimensions (T, C, Z, Y, X).
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

    Returns
    -------
    mask : xr.DataArray
        Binary mask with dimensions (T, Z, Y, X).
    labels : xr.DataArray
        Labeled regions with dimensions (T, Z, Y, X).

    Raises
    ------
    ValueError
        If threshold_method is not one of ['yen', 'li'].

    """
    if threshold_method not in threshold_method_choices:
        msg = f"threshold_method must be one of {threshold_method_choices}"
        raise ValueError(msg)

    # Compute geometric average across channels
    if len(channels) == 1:
        ga = im.sel(C=channels[0])
    else:
        ga = im.sel(C=list(channels)).astype(float).prod(dim="C") ** (1 / len(channels))

    # Apply wiener filter if requested
    ga_wiener = ga
    if wiener:
        ga_wiener = xr.apply_ufunc(
            _wiener_2d,
            ga,
            input_core_dims=[["Y", "X"]],
            output_core_dims=[["Y", "X"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[ga.dtype],
        )

    # Apply thresholding
    mask = xr.apply_ufunc(
        _threshold_2d,
        ga_wiener,
        kwargs={"method": threshold_method},
        input_core_dims=[["Y", "X"]],
        output_core_dims=[["Y", "X"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
    )

    # Remove small objects if requested
    if min_size:
        mask = xr.apply_ufunc(
            _remove_small_2d,
            mask,
            kwargs={"min_size": min_size},
            input_core_dims=[["Y", "X"]],
            output_core_dims=[["Y", "X"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[bool],
        )

    # Apply binary closing
    mask = xr.apply_ufunc(
        _closing_2d,
        mask,
        input_core_dims=[["Y", "X"]],
        output_core_dims=[["Y", "X"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
    )

    # Clear border if requested
    if clear_border:
        mask = xr.apply_ufunc(
            _clear_border_2d,
            mask,
            input_core_dims=[["Y", "X"]],
            output_core_dims=[["Y", "X"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[bool],
        )

    # Label connected components
    labels = xr.apply_ufunc(
        _label_2d,
        mask,
        input_core_dims=[["Y", "X"]],
        output_core_dims=[["Y", "X"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
    )

    if watershed:
        return process_watershed(
            im,
            mask,
            labels,
            channels=channels,
            randomwalk=randomwalk,
        )

    return mask, labels


def process_watershed(
    d_im: xr.DataArray,
    mask: xr.DataArray,
    labels: xr.DataArray,
    channels: tuple[str, ...],
    *,
    randomwalk: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Apply watershed to xarray DataArray.

    Parameters
    ----------
    d_im : xr.DataArray
        Input image.
    mask : xr.DataArray
        Binary mask.
    labels : xr.DataArray
        Labeled regions.
    channels : tuple[str, ...]
        Channel names (first channel used for intensity).
    randomwalk : bool, optional
        Use random walker instead of watershed (default=False).

    Returns
    -------
    mask : xr.DataArray
        Updated mask (not modified in current implementation, returned as is).
    labels : xr.DataArray
        Updated labels after watershed.

    """

    def apply_watershed_2d(
        dist: NDArray[Any],
        lbl: NDArray[np.int32],
        msk: NDArray[np.bool_],
        intensity: NDArray[Any],
    ) -> NDArray[np.int32]:
        # dist is distance transform of mask
        # lbl is initial labels
        # msk is binary mask

        pr = skmeasure.regionprops(lbl, intensity_image=intensity)  # type: ignore[no-untyped-call]
        if not pr:
            return lbl

        max_diameter = pr[0].equivalent_diameter_area
        for p in pr[1:]:
            max_diameter = max(max_diameter, p.equivalent_diameter_area)

        size = max_diameter * 2.20

        coords = feature.peak_local_max(  # type: ignore[no-untyped-call]
            dist,
            labels=lbl,
            footprint=np.ones((int(size), int(size))),
            min_distance=int(size),
            exclude_border=False,
        )
        local_maxi = np.zeros_like(dist, dtype=bool)
        local_maxi[tuple(coords.T)] = True

        markers = skmeasure.label(local_maxi)  # type: ignore[no-untyped-call]

        if randomwalk:
            markers[~msk] = -1
            labels_ws = segmentation.random_walker(msk, markers, mode="bf")
        else:
            labels_ws = segmentation.watershed(-dist, markers, mask=lbl)  # type: ignore[no-untyped-call]

        return labels_ws.astype(np.int32)  # type: ignore[no-any-return]

    # We need intensity image of channel 0.
    ch0 = d_im.sel(C=channels[0])

    def wrapper(
        msk: NDArray[np.bool_], lbl: NDArray[np.int32], intensity: NDArray[Any]
    ) -> NDArray[np.int32]:
        dist = ndimage.distance_transform_edt(msk)
        return apply_watershed_2d(dist, lbl, msk, intensity)

    new_labels = xr.apply_ufunc(
        wrapper,
        mask,
        labels,
        ch0,
        input_core_dims=[["Y", "X"], ["Y", "X"], ["Y", "X"]],
        output_core_dims=[["Y", "X"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[np.int32],
    )

    return mask, new_labels


def ratio(
    im: xr.DataArray,
    channels: tuple[str, str] = ("C", "R"),
    radii: Sequence[int] = (7, 3),
    mask: xr.DataArray | None = None,
) -> xr.DataArray:
    """Compute ratio image for xarray.DataArray.

    Parameters
    ----------
    im : xr.DataArray
        Input image.
    channels : tuple[str, str], optional
        Names for the two channels (Numerator, Denominator) (default=('C', 'R')).
    radii : Sequence[int], optional
        Each element contain a radius value for a median filter cycle (default=(7, 3)).
    mask : xr.DataArray | None, optional
        Binary mask to apply to the ratio image.

    Returns
    -------
    xr.DataArray
        The ratio image.

    """
    num = im.sel(C=channels[0])
    den = im.sel(C=channels[1])

    # Compute ratio, handling division by zero and NaNs
    ratio = num / den
    ratio = ratio.where(np.isfinite(ratio), 0)

    # Apply median filters if requested
    if radii:

        def apply_median_filters(
            im: NDArray[Any], radii_seq: Sequence[int]
        ) -> NDArray[Any]:
            filtered = im
            for radius in radii_seq:
                filtered = ndimage.median_filter(filtered, radius)
            return filtered

        ratio = xr.apply_ufunc(
            apply_median_filters,
            ratio,
            kwargs={"radii_seq": radii},
            input_core_dims=[["Y", "X"]],
            output_core_dims=[["Y", "X"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[ratio.dtype],
        )

    # Apply mask if provided
    if mask is not None:
        ratio = ratio * mask

    return ratio


def measure(  # noqa: PLR0913
    im: xr.DataArray,
    labels: xr.DataArray,
    channels: Sequence[str] = ("C", "G", "R"),
    channels_cl: tuple[str, str] = ("C", "R"),
    channels_ph: tuple[str, str] = ("G", "C"),
    radii: Sequence[int] | None = None,
    *,
    ratios_from_image: bool = True,
) -> tuple[dict[int, pd.DataFrame], dict[str, list[list[Any]]]]:
    """Measure properties for xarray.DataArray.

    Parameters
    ----------
    im : xr.DataArray
        Input image.
    labels : xr.DataArray
        Labeled image.
    channels : Sequence[str], optional
        All channels (default=('C', 'G', 'R')).
    channels_cl : tuple[str, str], optional
        Numerator and denominator channels for cl ratio (default=('C', 'R')).
    channels_ph : tuple[str, str], optional
        Numerator and denominator channels for pH ratio (default=('G', 'C')).
    radii : Sequence[int] | None, optional
        Radii of the optional median average performed on ratio images (default=None).
    ratios_from_image : bool, optional
        Boolean for executing ratio i.e. compute ratio images (default=True).

    Returns
    -------
    meas : dict[int, pd.DataFrame]
        For each label in labels: {'label': df}.
        DataFrame columns are: mean intensity of all channels,
        'equivalent_diameter', 'eccentricity', 'area', ratios from the mean
        intensities and optionally ratios from ratio-image.
    pr : dict[str, list[list[Any]]]
        For each channel: {'channel': [props]} i.e. {'channel': [time][label]}.

    """
    pr: dict[str, list[list[Any]]] = defaultdict(list)
    # Ensure dimensions order for iteration (T, ...)
    # If im has Z, we expect labels to have Z.
    # We iterate over T.

    times = im.coords["T"].to_numpy()

    for ch in channels:
        pr[ch] = []
        for t in range(len(times)):
            im_frame = im.sel(C=ch).isel(T=t)
            lbl_frame = labels.isel(T=t)

            # Convert to numpy for regionprops
            im_np = im_frame.to_numpy()
            lbl_np = lbl_frame.to_numpy()

            props = skmeasure.regionprops(lbl_np, intensity_image=im_np)  # type: ignore[no-untyped-call]
            pr[ch].append(props)

    meas: dict[int, pd.DataFrame] = {}
    # labels unique values (0 is background)
    # We can get unique labels from the labels DataArray (could be slow if large)
    # Alternatively, we iterate based on what regionprops found.
    # But we need consistent label IDs across time.
    unique_labels = np.unique(labels.compute().to_numpy())
    unique_labels = unique_labels[unique_labels > 0]

    for lbl in unique_labels:
        idx = []
        d = defaultdict(list)
        for t, _ in enumerate(times):
            # Check properties for channel 0 to see if label exists
            props = pr[channels[0]][t]
            try:
                # Find the prop with this label
                # This search might be inefficient for many labels
                prop = next(p for p in props if p.label == lbl)
                idx.append(t)
                d["equivalent_diameter"].append(prop.equivalent_diameter_area)
                d["eccentricity"].append(prop.eccentricity)
                d["area"].append(prop.area)

                # Get mean intensity for all channels
                # We assume consistent ordering/existence of props across channels
                # But safer to find by label again or assume same index if sorted
                # regionprops returns properties sorted by label.
                # So if label exists in ch0, it exists in others (same label image)
                i_label = [p.label for p in props].index(lbl)

                for ch in channels:
                    d[ch].append(pr[ch][t][i_label].intensity_mean)

            except StopIteration:
                pass  # Label absent at this time

        res_df = pd.DataFrame({k: np.array(v) for k, v in d.items()}, index=idx)
        res_df["r_cl"] = res_df[channels_cl[0]] / res_df[channels_cl[1]]
        res_df["r_pH"] = res_df[channels_ph[0]] / res_df[channels_ph[1]]
        meas[int(lbl)] = res_df

    if ratios_from_image:
        # Calculate ratio images (DataArray)
        r_cl_da = ratio(
            im,
            channels=channels_cl,
            radii=radii if radii is not None else (7, 3),
            mask=labels > 0,
        )
        r_ph_da = ratio(
            im,
            channels=channels_ph,
            radii=radii if radii is not None else (7, 3),
            mask=labels > 0,
        )

        r_ph_list = []
        r_cl_list = []

        # Iterate time and calculate median within labels
        for t in range(len(times)):
            r_ph_frame = r_ph_da.isel(T=t).to_numpy()
            r_cl_frame = r_cl_da.isel(T=t).to_numpy()
            lbl_frame_np = labels.isel(T=t).to_numpy()

            # ndimage.median works with labels and index list
            r_ph_list.append(
                ndimage.median(r_ph_frame, lbl_frame_np, index=unique_labels)
            )
            r_cl_list.append(
                ndimage.median(r_cl_frame, lbl_frame_np, index=unique_labels)
            )

        ratios_ph = np.array(r_ph_list)
        ratios_cl = np.array(r_cl_list)

        for i, lbl in enumerate(unique_labels):
            # For xarray implementation, let's be safer.
            # We have ratios_ph[:, i]. This corresponds to unique_labels[i].

            # We need to align with meas[lbl] which has specific time indices.

            res_df = pd.DataFrame(
                {
                    "r_pH_median": ratios_ph[:, i],
                    "r_cl_median": ratios_cl[:, i],
                },
                index=range(len(times)),  # Default index 0..T-1
            )
            # meas[lbl] has index=idx (subset of times).
            # Join inner will select matching times.
            meas[int(lbl)] = pd.concat([meas[int(lbl)], res_df], axis=1, join="inner")

    return meas, pr


def d_plot_meas(
    bgs: pd.DataFrame, meas: dict[int, pd.DataFrame], channels: Sequence[str]
) -> Figure:
    """Plot meas object.

    Plot r_pH, r_cl, mean intensity for each channel and estimated bg over
    timepoints for each label (color coded).

    Parameters
    ----------
    bgs : pd.DataFrame
        Estimated bg returned from bg()
    meas : dict[int, pd.DataFrame]
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
    axes = cast("np.ndarray[Any, Any]", fig.subplots(nrows, ncols))
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
    img: ImFrame,
    title: str | None = None,
    hpix: pd.DataFrame | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Figure:
    """Summary graphics for Flat-Bias images.

    Parameters
    ----------
    img : ImFrame
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
        im: ImSequence,
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
        if vmin is None:
            vmin = float(np.percentile(im, 18.4))  # 1/e (66.6 %)
        elif vmax is None:
            vmax = float(np.percentile(im, 81.6))  # 1/e (66.6 %)
        img = ax.imshow(im, vmin=vmin, vmax=vmax, cmap="turbo")
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


def plt_img_profile_2(img: ImFrame, title: str | None = None) -> Figure:
    """Summary graphics for Flat-Bias images.

    Parameters
    ----------
    img : ImFrame
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
    vmin, vmax = [float(val) for val in np.percentile(img, [18.4, 81.6])]  # 66.6 %
    ax.imshow(img, vmin=vmin, vmax=vmax, cmap="turbo")
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


def hotpixels(bias: ImFrame, n_sd: int = 20) -> pd.DataFrame:
    """Identify hot pixels in a bias-dark frame.

    After identification of first outliers recompute masked average and std
    until convergence.

    Parameters
    ----------
    bias : ImFrame
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
        m_ave = np.ma.masked_array(bias, m).mean()
        m_std = np.ma.masked_array(bias, m).std()
        m = bias > m_ave + n_sd * m_std
        if n_hpix == m.sum():
            break
        n_hpix = m.sum()
    w = np.where(m)
    hpix_df = pd.DataFrame({"y": w[0], "x": w[1]})
    return hpix_df.assign(val=lambda row: bias[row.y, row.x])


def correct_hotpixel(
    img: ImFrame, y: int | NDArray[np.int_], x: int | NDArray[np.int_]
) -> None:
    """Correct hot pixels in a frame.

    Substitute indicated position y, x with the median value of the 4 neighbor
    pixels.

    Parameters
    ----------
    img : ImFrame
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


def read_tiffmd(fp: Path, channels: Sequence[str]) -> tuple[Array, Metadata]:
    """Read multichannel TIFF timelapse image."""
    n_channels = len(channels)
    img = io.read_image(fp, channels)
    data = img.data
    # Squeeze Z dimension if singleton, for backward compatibility
    if img.sizes["Z"] == 1:
        data = da.squeeze(data, axis=2)  # axis 2 is Z in TCZYX

    # Get the pre-attached Metadata object from attrs
    md = img.attrs["metadata"]
    if md.size_c[0] % n_channels:
        msg = "n_channel mismatch total length of TIFF sequence"
        raise ValueError(msg)
    return data.astype(np.int32), md
