"""Remaining cli supporting functions."""
import os
import sys
import zipfile
from typing import Any, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import ndimage  # type: ignore
from skimage import io  # type: ignore

from nimg import nimg
from nimg.nimg import ImArray

mpl.rcParams["figure.max_open_warning"] = 199  # type: ignore
methods_bg = ("entropy", "arcsinh", "adaptive", "li_adaptive", "li_li")
methods_fg = ("yen", "li")


def dark(fp: str, thr: int = 95) -> Tuple[ImArray, pd.DataFrame, plt.Figure]:
    """Estimate image for dark correction.

    Read zip; median z-project; median filter(1).

    Parameters
    ----------
    fp : str
        File name to be processed.
    thr : float
        Threshold for hot pixels calculation.

    Returns
    -------
    imf : np.array
        Filtered image; pre0serve dtype of input im.
    df_hp: pd.DataFrame
        Coordinates (x,y) and values for all hotpixels.
    f : plt.figure
        Plot of the dark image and its histogram.

    """
    im = zipread(fp)
    print(im.shape)  # Reads only the first YX plane currently.
    zp = nimg.zproject(im)
    # imf = ni.im_median(zp)
    imf = ndimage.median_filter(
        im, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
    )
    f = plt.figure(figsize=(6.75, 9.25))
    f.suptitle("DARK stack")
    #
    with plt.style.context("seaborn-ticks"):  # type: ignore
        plt.subplot(321)  # type: ignore
        plt.hist(imf.ravel(), bins=256, histtype="step", lw=4)  # type: ignore
        plt.yscale("log")  # type: ignore
        plt.title("DARK image")
    #
    plt.subplot(322)  # type: ignore
    plt.imshow(imf, cmap=plt.cm.inferno_r)  # type: ignore
    plt.colorbar()  # type: ignore
    plt.axis("off")  # type: ignore
    plt.title("exported DARK image")
    #
    with plt.style.context("seaborn-ticks"):  # type: ignore
        plt.subplot(323)  # type: ignore
        plt.hist(im.ravel(), bins=256, histtype="step", lw=4)  # type: ignore
        plt.yscale("log")  # type: ignore
        plt.title("original stack")
    #
    plt.subplot(324)  # type: ignore
    # hot pixels; cast to float because uint screwed up to range max
    d = imf.astype(float) - zp.astype(float)
    thr = np.std(d) * thr
    hot_pixels = np.nonzero(abs(d) > thr)
    df_hp = pd.DataFrame(
        {"row": hot_pixels[0], "col": hot_pixels[1], "val": zp[hot_pixels]}
    )
    plt.imshow(zp)  # type: ignore
    plt.plot(hot_pixels[1], hot_pixels[0], "r+", mfc="none", mec="w", ms=18)
    plt.colorbar()  # type: ignore
    plt.axis("off")  # type: ignore
    plt.title("projected stack")
    #
    f.tight_layout()
    return imf, df_hp, f


def flat(
    fflat: str, fdark: str, method: str = "overall"
) -> Tuple[NDArray[np.float_], plt.Figure]:
    """Estimate image for flat correction.

    First subtract the dark, then apply normalization either at each plane 'single'
    or a 'overall' (default) median projection of all planes.

    Parameters
    ----------
    fflat : str
        File name of the stack to be processed.
    fdark : str
        File name of the dark image.
    method : {'overall', 'single'}, optional
        Choices in brackets, default first.

    Returns
    -------
    flat : np.array
        Flat image.
    f : plt.figure
        Plot of the flat image and its histogram.

    """
    # read files
    dark = io.imread(fdark, plugin="tifffile")
    im = zipread(fflat)
    if not im.shape[1:] == dark.shape:
        # TODO Raise exception
        sys.exit("flat images serie and dark image size mismatch")
    ims = im - dark
    if method == "overall":
        flat = np.median(ims, axis=0)
        # flat = ni.im_median(flat)
        flat = ndimage.median_filter(
            im, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        )
        flat = flat / np.mean(flat)
    if method == "single":
        ims = ims.astype(float)
        for i, im in enumerate(ims):
            # ims[i] = ni.im_median(im)
            ims[i] = ndimage.median_filter(
                im, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            )
            ims[i] = ims[i] / np.mean(ims[i])
        flat = np.median(ims, axis=0)
    # Pdf output
    f = plt.figure(figsize=(6.75, 9.25))
    f.suptitle("FLAT stack")
    # table
    ax = plt.subplot2grid((6, 4), (0, 0), colspan=4)  # type: ignore
    ax.set_axis_off()
    # http://nipunbatra.github.io/2014/08/latexify/
    params = {
        # 'axes.labelsize': 8,  # fontsize for x and y labels
        "font.size": 9,
        "font.family": "serif",
    }
    mpl.rcParams.update(params)  # type: ignore
    fcommon, fdark_relative, fflat_relative = common_path(fdark, fflat)
    data = pd.Series(
        data=[fcommon, fdark_relative, fflat_relative],
        index=pd.Index(["root", "dark", "flat"]),
        name="Files",  # type: ignore
    )
    pd.tools.plotting.table(ax=ax, data=data, loc=3)  # type: ignore
    mpl.rcdefaults()  # type: ignore
    # FLAT
    ax0 = plt.subplot2grid((6, 4), (1, 0), colspan=3, rowspan=3)  # type: ignore
    plt.axis("off")  # type: ignore
    img0 = ax0.imshow(flat)
    ax0.set_title("exported FLAT image")
    ax1 = plt.subplot2grid((6, 4), (1, 3), colspan=1, rowspan=3)  # type: ignore
    plt.axis("off")  # type: ignore
    plt.colorbar(img0, ax=ax1, fraction=0.9, shrink=0.78, aspect=14)  # type: ignore
    #
    # hist flat
    with plt.style.context("seaborn-ticks"):  # type: ignore
        plt.subplot2grid((6, 4), (4, 0), colspan=2, rowspan=2)  # type: ignore
        plt.hist(flat.ravel(), bins=256, histtype="step", lw=2, color="crimson")  # type: ignore
        plt.yscale("log")  # type: ignore
        plt.grid()
        plt.title("FLAT image")
    # hist stack
    with plt.style.context("seaborn-ticks"):  # type: ignore
        plt.subplot2grid((6, 4), (4, 2), colspan=2, rowspan=2)  # type: ignore
        plt.hist(im.ravel(), bins=256, histtype="step", lw=2, color="grey")  # type: ignore
        plt.yscale("log")  # type: ignore
        plt.ylim(
            0.1,
        )
        plt.grid()
        plt.title("original stack")
    #
    f.tight_layout()
    return flat, f


def common_path(path1: str, path2: str) -> Tuple[str, str, str]:
    """Find common absolute path.

    Utility function to find common absolute path.

    Parameters
    ----------
    path1, path2 : string
        Two file paths.

    Returns
    -------
    3-tupla
        (common absolute path, relative path 1, relative path2)
        Relative paths with respect to the common path.

    Examples
    --------
    >>> common_path('/home/dan/docs/arte/unimi/grafEPS.tgz', '/home/dati/GBM_persson/')
    ('/home', 'dan/docs/arte/unimi/grafEPS.tgz', 'dati/GBM_persson')

    """
    fcommon = os.path.commonprefix([os.path.abspath(path1), os.path.abspath(path2)])
    fcommon = os.path.dirname(fcommon)  # stop at the directory in common path
    f1 = os.path.relpath(path1, start=fcommon)
    f2 = os.path.relpath(path2, start=fcommon)
    return fcommon, f1, f2


def zipread(fp: str) -> Any:
    """Unzip and read a single TIF.

    Return the image (np.array).

    Parameters
    ----------
    fp : str
        File name to be processed.

    Returns
    -------
     : np.array
        Flat image.

    """
    with zipfile.ZipFile(fp) as myzip:
        with myzip.open(myzip.filelist[0]) as myfile:
            return io.imread(myfile)
