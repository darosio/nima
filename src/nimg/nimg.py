"""Main library module.

Contains functions for the analysis of multichannel timelapse images. It can
be used to apply dark, flat correction; segment cells from bg; label cells;
obtain statistics for each label; compute ratio and ratio images between
channels.
"""
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import skimage
import skimage.feature
import skimage.segmentation
import skimage.transform
import tifffile
from scipy import ndimage, signal
from skimage import filters
from skimage.morphology import disk


def im_print(im, verbose=False):
    """Print useful information about im."""
    print(
        "ndim = ",
        im.ndim,
        "| shape = ",
        im.shape,
        "| max = ",
        im.max(),
        "| min = ",
        im.min(),
        "| size = ",
        im.size,
        "| dtype = ",
        im.dtype,
    )
    if verbose:
        if im.ndim == 3:
            for i, image in enumerate(im):
                print(
                    "i = {0:4d} | size = {1} | zeros = {2}".format(
                        i, image.size, np.count_nonzero(image == 0)
                    )
                )


def myhist(im, bins=60, log=False, nf=0):
    """Plot image intensity as histogram.

    ..note:: Consider deprecation.

    """
    # sns.set_style('ticks', {'axes.grid': True})
    # sns.pointplot(imf.flatten(), kde=False,
    #               hist_kws={"histtype": "step", "linewidth": 3})
    hist, bin_edges = np.histogram(im, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    if nf:
        plt.figure()
    plt.plot(bin_centers, hist, lw=2)
    if log:
        plt.yscale("log")


def plot_im_series(im, cmap=plt.cm.gray, horizontal=True, **kw):
    """Plot a image series with a maximum of 9 elements.

    ..note:: Consider deprecation. Use d_show() instead.

    """
    if horizontal:
        plt.figure(figsize=(12, 5.6))
        s = 100 + len(im) * 10 + 1
    else:
        plt.figure(figsize=(5.6, 12))
        s = len(im) * 100 + 10 + 1
    for i, img in enumerate(im):
        plt.subplot(s + i)
        plt.imshow(img, cmap=cmap, **kw)
        plt.axis("off")
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)


def plot_otsu(im, cmap=plt.cm.gray):
    """Otsu threshold and plot im_series.

    .. note:: Consider deprecation.

    """
    val = filters.threshold_otsu(im)
    mask = im > val
    plot_im_series(im * mask, cmap=cmap)
    return mask


def im_median(im, radius=0, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])):
    """Image median filter.

    Return the median filtered image *im*
    plane by plane with e.g. radius=3 calculate 3D median filter.

    Parameters
    ----------
    im : np.array
        either 2D or 3D image
    radius : float, optional
        Do 3D filtering as it simply uses ndimage.filters.median_filter
        (default 0: do not use 3D but 2D filtering with footprint).
    footprint : np.array, optional
        default is equivalent to skimage.morphology.disk(1) and to median
        filter of Fiji/ImageJ with radius=0.5.

    Returns
    -------
    np.array
        Filtered image; preserve dtype of input im.

    Examples
    --------
    >>> im = np.indices([3, 3]).sum(axis=0)
    >>> im
    array([[0, 1, 2],
           [1, 2, 3],
           [2, 3, 4]])
    >>> im_median(im, footprint=np.ones((5, 5)))
    array([[2, 2, 2],
           [2, 2, 2],
           [2, 2, 2]])

    """
    filter = ndimage.filters.median_filter
    if radius:
        return filter(im, size=radius)
    else:
        if im.ndim == 2:
            return filter(im, footprint=footprint)
        else:
            imf = np.zeros(im.shape).astype(im.dtype)
            for i, img in enumerate(im):
                imf[i] = filter(img, footprint=footprint)
            return imf


def zproject(im, func=np.median):
    """Perform z-projection of a 3D image.

    func must support axis= and out= API like np.median, np.mean, np.percentile


    Parameters
    ----------
    im : np.array
        Image (pln, row, col).

    Returns
    -------
    np.array(row, col)
        2D (projected) image (median by default). Preserve dtype of input.

    Raises
    ------
    AssertionError
        If the input image is not 3D.

    """
    assert (
        im.ndim == 3 and len(im) == im.shape[0]
    ), "Input must be 3D-grayscale (pln, row, col)"
    # maintain same dtype as input im; odd and even
    zproj = np.zeros(im.shape[1:]).astype(im.dtype)
    func(im[1:], axis=0, out=zproj)
    return zproj


def read_tiff(fp, channels):
    """Read multichannel tif timelapse image.

    Parameters
    ----------
    fp : path
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
    im = tifffile.imread(fp)
    n_channels = len(channels)
    if len(im) % n_channels:
        raise Exception("n_channel mismatch total lenght of tif sequence")
    else:
        d_im = {}
        for i, ch in enumerate(channels):
            d_im[ch] = im[i::n_channels]
        return d_im, n_channels, len(im) // n_channels


def d_show(d_im, **kws):
    """Imshow for dictionary of image (d_im). Support plt.imshow kws."""
    MAX_ROWS = 9
    n_channels = len(d_im.keys())
    first_channel = d_im[list(d_im.keys())[0]]
    if first_channel.ndim == 2:
        n_times = 1
    elif first_channel.ndim == 3:
        n_times = len(first_channel)
    if n_times <= MAX_ROWS:
        rng = range(n_times)
        n_rows = n_times
    else:
        step = np.ceil(n_times / MAX_ROWS).astype(int)
        rng = range(0, n_times, step)
        n_rows = len(rng)

    f = plt.figure(figsize=(16, 16))
    for n, ch in enumerate(sorted(d_im.keys())):
        for i, r in enumerate(rng):
            plt.subplot(n_rows, n_channels, i * n_channels + n + 1)
            if n_times == 1:
                img0 = plt.imshow(d_im[ch], **kws)
            else:
                img0 = plt.imshow(d_im[ch][r], **kws)
            plt.colorbar(img0, orientation="vertical", pad=0.02, shrink=0.85)
            plt.xticks([])
            plt.yticks([])
            plt.ylabel(ch + " @ t = " + str(r))
    plt.subplots_adjust(wspace=0.2, hspace=0.02, top=0.9, bottom=0.1, left=0, right=1)
    return f


# median
def d_median(d_im):
    """Median filter on dictionary of image (d_im). Return a new d_im copy."""
    d_out = {}
    for k, im in d_im.items():
        d_out[k] = im_median(im)
    return d_out


def d_shading(d_im, dark, flat, clip=True):
    """Shading correction on d_im.

    Subtract dark; then divide by flat.

    Works either with flat or d_flat
    Need also dark for each channel because it can be different when using
    different acquisiton times.

    Parameters
    ----------
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
    raise_msg = "Unexpected imput"
    assert type(dark) == np.ndarray or dark.keys() == d_im.keys(), raise_msg
    assert type(flat) == np.ndarray or flat.keys() == d_im.keys(), raise_msg
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


def bg(im, kind="arcsinh", perc=10, radius=10, adaptive_radius=None, arcsinh_perc=80):
    """Bg segmentation.

    Return median, whole vector, figures (in a [list])

    Parameters
    ----------
    kind : {'arcsinh', 'entropy', 'adaptive', 'li_adaptive', 'li_li'}
        Method used for the segmentation.
    perc : int, optional
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
        adaptive_radius = im.shape[1] / 2
        if adaptive_radius % 2 == 0:  # 0.12.0 check for even number
            adaptive_radius += 1
    if perc < 0 or perc > 100:
        raise Exception("perc must be in [0, 100] range")
    else:
        perc /= 100
    if kind == "arcsinh":
        lim = np.arcsinh(im)
        lim = ndimage.percentile_filter(lim, arcsinh_perc, size=radius)
        lim_ = True
        title = radius, perc
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
        # m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        f = imm > filters.threshold_local(imm, adaptive_radius)
        m = ~f * m
    elif kind == "li_li":
        lim_ = False
        title = None
        li = filters.threshold_li(im.copy())
        m = im < li
        # m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        # To avoid zeros generated after first thesholding, clipping to the
        # min value of original image is needed before second thesholding.
        thr2 = filters.threshold_li(imm.clip(np.min(im)))
        m = im < thr2
        # ###mm = skimage.morphology.binary_closing(mm)
    pixel_values = im[m]
    iqr = np.percentile(pixel_values, [25, 50, 75])
    #
    f1 = plt.figure(figsize=(9, 5))
    plt.subplot(121)
    masked = im * m
    img0 = plt.imshow(masked, cmap=plt.cm.inferno)
    plt.colorbar(img0, orientation="horizontal")
    plt.title(kind + " " + str(title) + "\n" + str(iqr))
    #
    plt.subplot(122)
    myhist(im[m], log=True)
    plt.tight_layout()

    if lim_:
        f2 = plt.figure(figsize=(9, 4))
        plt.subplot(131)
        img0 = plt.imshow(lim)
        plt.colorbar(img0, orientation="horizontal")
        #
        plt.subplot(132)
        myhist(lim)
        #
        host = plt.subplot(133)
        # plot bg vs. perc
        ave, sd, median = ([], [], [])
        delta = lim.max() - lim.min()
        delta /= 2
        rng = np.linspace(lim.min() + delta / 20, lim.min() + delta, 20)
        par = host.twiny()
        # plt.make_patch_spines_invisible(par)
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
        f2.tight_layout()
        return iqr[1], pixel_values, [f1, f2]
    else:
        return iqr[1], pixel_values, [f1]


def d_bg(d_im, downscale=None, kind="li_adaptive", clip=True, **kw):
    """Bg segmentation for d_im.

    Parameters
    ----------
    d_im : d_im
        desc
    downscale : {None, tupla}
        Tupla, x, y are downscale factors for rows, cols.
    kind : {'li_adaptive', 'arcsinh', 'entropy', 'adaptive', 'li_li'}
        Bg method.
    clip : bool
        Boolean (default=True) for clipping values >=0.
    **kw : dict
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
    d_bg_values :

    """
    d_bg = {}
    d_bg_values = {}
    d_cor = {}
    d_fig = {}
    for k in d_im.keys():
        d_bg[k] = []
        d_bg_values[k] = []
        d_cor[k] = []
        d_fig[k] = []
        for t, im in enumerate(d_im[k]):
            if downscale:
                im = skimage.transform.downscale_local_mean(im, downscale)
            med, v, ff = bg(im, kind, **kw)
            d_bg[k].append(med)
            d_bg_values[k].append(v)
            d_cor[k].append(d_im[k][t] - med)
            d_fig[k].append(ff)
        d_cor[k] = np.array(d_cor[k])
    if clip:
        for k in d_cor.keys():
            d_cor[k] = d_cor[k].clip(0)
    bgs = pd.DataFrame(d_bg)
    return d_cor, bgs, d_fig, d_bg_values


def d_mask_label(
    d_im,
    min_size=640,
    channels=["C", "G", "R"],
    threshold_method="yen",
    wiener=False,
    watershed=False,
    clear_border=False,
    randomwalk=False,
):
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

    Returns
    -------
    None

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
    d_im["mask"] = mask
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


def d_ratio(d_im, name="r_cl", channels=["C", "R"], radii=(7, 3)):
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
    name : string
        Name (default='r_cl') for the new key.
    channels : list of string
        Names (default=['C', 'R']) for the two channels [Numerator,
        Denominator].
    radii : tupla of int, optional
        Each element contain a radius value for a median filter cycle.

    Returns
    -------
    None

    Add a key named "name" and containing the calculated ratio to d_im.

    """
    with np.errstate(divide="ignore", invalid="ignore"):
        # 0/0 and num/0 can both happen.
        ratio = d_im[channels[0]] / d_im[channels[1]]
    for i, r in enumerate(ratio):
        ratio[i] = pd.DataFrame(r).replace([-np.inf, np.nan, np.inf], 0)
        for radius in radii:
            ratio[i] = ndimage.median_filter(ratio[i], radius)
        ratio[i] *= d_im["mask"][i]
    d_im[name] = ratio


def d_meas_props(
    d_im,
    channels=["C", "G", "R"],
    channels_cl=["C", "R"],
    channels_pH=["G", "C"],
    ratios_from_image=True,
    radii=None,
):
    """Calculate pH and cl ratios and labelprops.

    Parameters
    ----------
    d_im : d_im
        desc
    name : string
        Name (default='r_cl') for the new key.
    channels : list of string
        All d_im channels (default=['C', 'G', 'R']).
    channels_cl : list of string
        Names (default=['C', 'R']) for the two channels [Numerator,
        Denominator] for cl ratio.
    channels_pH : list of string
        Names (default=['G', 'C']) for the two channels [Numerator,
        Denominator] for pH ratio.
    ratios_from_image : bool, optional
        Boolean (default=True) for executing d_ratio i.e. compute ratio images.

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
    pr = {}
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
        d = {ch: [] for ch in channels}
        d["equivalent_diameter"] = []
        d["eccentricity"] = []
        d["area"] = []
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
        df = pd.DataFrame(d, index=idx)
        df["r_cl"] = df[channels_cl[0]] / df[channels_cl[1]]
        df["r_pH"] = df[channels_pH[0]] / df[channels_pH[1]]
        meas[label] = df

    if ratios_from_image:
        kwargs = {}
        if radii:
            kwargs["radii"] = radii
        d_ratio(d_im, "r_cl", channels=channels_cl, **kwargs)
        d_ratio(d_im, "r_pH", channels=channels_pH, **kwargs)
        r_pH = []
        r_cl = []
        for time, (pH, cl) in enumerate(zip(d_im["r_pH"], d_im["r_cl"])):
            r_pH.append(ndimage.median(pH, d_im["labels"][time], index=labels))
            r_cl.append(ndimage.median(cl, d_im["labels"][time], index=labels))
        ratios_pH = np.array(r_pH)
        ratios_cl = np.array(r_cl)
        for label in meas:
            df = pd.DataFrame(
                {
                    "r_pH_median": ratios_pH[:, label - 1],
                    "r_cl_median": ratios_cl[:, label - 1],
                }
            )
            # concat only on index that are present in both
            meas[label] = pd.concat([meas[label], df], axis=1, join="inner")

    return meas, pr


def d_plot_meas(bgs, meas, channels):
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
    fig : plt.fig
        Figure.

    """
    colors = ["k", "b", "g", "r", "y", "c", "m"]
    NCOLS = 2
    n_axes = len(channels) + 3  # 2 ratios and 1 bg axes
    nrows = int(np.ceil(n_axes / NCOLS))
    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 5, nrows * 3))
    # r_pH with legend
    legend = []
    for k, df in meas.items():
        legend.append(k)
        color = colors[(k - 1) % len(colors)]
        df["r_pH"].plot(marker="o", color=color, ax=axes[0, 0])
        df["r_cl"].plot(marker="o", color=color, ax=axes[0, 1])
    for k, df in meas.items():
        color = colors[(k - 1) % len(colors)]
        if "r_pH_median" in df:
            df["r_pH_median"].plot(style="--", color=color, lw=2, ax=axes[0, 0])
        if "r_cl_median" in df:
            df["r_cl_median"].plot(style="--", color=color, lw=2, ax=axes[0, 1])
    axes[0, 0].set_ylabel("r_pH")
    axes[0, 0].grid()
    axes[0, 1].set_ylabel("r_cl")
    axes[0, 1].grid()
    axes[0, 0].set_title("pH")
    axes[0, 1].set_title("Cl")
    axes[0, 0].legend(legend)

    for n, ch in enumerate(channels, 2):
        i = n // NCOLS
        j = n % NCOLS  # * 2
        for k, df in meas.items():
            color = colors[(k - 1) % len(colors)]
            df[ch].plot(marker="o", color=color, ax=axes[i, j])
        axes[i, j].set_title(ch)
        axes[i, j].grid()

    ch_colors = [
        i.lower() if i.lower() in matplotlib.colors.BASE_COLORS else "k"
        for i in bgs.columns
    ]
    if n_axes == nrows * NCOLS:
        axes.ravel()[-2].set_xlabel("time")
        axes.ravel()[-1].set_xlabel("time")
        bgs.plot(ax=axes[nrows - 1, NCOLS - 1], grid=True, color=ch_colors)
    else:
        axes.ravel()[-3].set_xlabel("time")
        axes.ravel()[-2].set_xlabel("time")
        bgs.plot(ax=axes[nrows - 1, NCOLS - 2], grid=True, color=ch_colors)
        ax = axes.ravel()[-1]
        plt.delaxes(ax)

    fig.tight_layout()
    return fig
