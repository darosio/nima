"""
Analyze a multichannel tiff stack e.g. [G, R, C, G2] in timelapse.

Create 'nimg' folder and for each TIFFSTK save:
    - bname_dim.png   dictionary image with segmentation.
    - bname_meas.png  plot of image bg and for each label values of channel
                      intensities and pH and Cl ratio.
    - bname/bg.csv    bg (value for each timepoint).
    - for each channel CH:
    - bname/bg-CH-MTH.pdf    Bg image and histogram for each timepoint.
    - for each label X:
    - bname/labelX.csv       Channels, ratios and properties values.
    - bname/labelX_rcl.tif   Image stack of the cl ratio for label X.
    - bname/labelX_rpH.tif   Image stack of the pH ratio for label X.

`nimg dark` read a stack of dark images (tiff-zip) and save (in current dir):
    - DARK image = median filter of median projection.
    - plot (histograms, median, projection, hot pixels).
    - csv file containing coordinates of hotpixels.

`nimg flat` read a stack of dark images (tiff-zip) and a DARK reference image,
and save (in current dir):
    - FLAT image = median filter of median projection.
    - plot (stack histograms, flat image and its histogram - name of stack and
      reference dark image)


New: DARK and FLAT are a single file, and both must be a d_im with appropriate
channels.


Usage:
  nimg dark <zipfile>
  nimg flat <zipfile> <darkfile>
  nimg [options] TIFFSTK [(-d DARK -f FLAT)] CHANNELS...
  nimg -h | --help
  nimg --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --silent      Do not print; verbose=0.
  -o OUT, --output OUT      Output folder path [default: nimg]
  --hotpixel, --hotpixels   Execute median filter with radius=0.5 to remove
                            hotpixel.
  -d DARK, --dark DARK      Dark for shading correction.
  -f FLAT, --flat FLAT      FLAT for shading correction.
  --bg-downscale X,Y        X and Y comma separeted (no spaces) for optional
                            downscaling.
  --bg-method MTH           Method for bg estimation [default: li_adaptive]
                            Available method: entropy, arcsinh, adaptive,
                            li_adaptive, li_li.

  --bg-radius R             Radius for entropy or arcsinh methods
                            [default: 10].
  --bg-adaptive-radius R    Radius for adaptive methods (default is calculated
                            as im.shape[1]/2).
  --bg-percentile P         Percentile for entropy or arcsinh methods
                            [default: 10].
  --bg-percentile-filter P  Percentile filter for arcsinh method [default: 80].

  --fg-method MTH           Method for fg estimation [default: yen]
                            Available method: li, yen.
  --min-size PIXELS         Minimum size for labeled objects [default: 2000].
  --clear-border            Remove labeled object touching image borders.
  --wiener                  Execute Wiener filter before segmentation.
  --watershed               Execute watershed on binary mask (to label cells).
  --randomwalk              Execute randomwalk on binary mask (to label cells).

  --channels-cl CH1/CH2     Channels for cl ratio (default is C/R).
  --channels-pH CH1/CH2     Channels for pH ratio (default is G/C).
  --no-image-ratios         Do not compute ratio images.
  --ratio-median-radii Rs   Tupla of values (default is 7, 3).
                            An integer value for each median filter pass.

"""
import os
import sys
import zipfile
from docopt import docopt
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage import io
import tifffile
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib as mpl
import nimg.nimg as ni
from nimg import __version__ as version

mpl.rcParams["figure.max_open_warning"] = 199
methods_bg = ("entropy", "arcsinh", "adaptive", "li_adaptive", "li_li")
methods_fg = ("yen", "li")


def main():
    args = docopt(__doc__, version=version)
    print(args)
    if args["dark"]:
        # parsing
        fzip = args["<zipfile>"]
        # computation
        # dark_im, dark_hotpixels, f = nimg.scripts.dark(fzip)
        dark_im, dark_hotpixels, f = dark(fzip)
        # output
        bname = "dark-" + os.path.splitext(os.path.basename(fzip))[0]
        f.savefig(bname + ".pdf")
        # TODO suppress UserWarning low contrast is actually expected here
        io.imsave(bname + ".tif", dark_im, plugin="tifffile")
        dark_hotpixels.to_csv(bname + ".csv")
        print("median [ IQR ] = ", np.median(dark_im), np.percentile(dark_im, [25, 75]))
    elif args["flat"]:
        # parsing
        fdark = args["<darkfile>"]
        fflat = args["<zipfile>"]
        # computation
        # flat_im, f = nimg.scripts.flat(fflat, fdark)
        flat_im, f = flat(fflat, fdark)
        # output
        bname = (
            "flat-"
            + os.path.splitext(os.path.basename(fflat))[0]
            + "-"
            + os.path.splitext(os.path.basename(fdark))[0]
        )
        f.savefig(bname + ".pdf")
        io.imsave(bname + ".tif", flat_im, plugin="tifffile")
    else:
        # parsing
        channels = args["CHANNELS"]
        # methods
        method_fg = args["--fg-method"]
        if method_fg not in methods_fg:
            print(__doc__)
            print("METHOD must be one of: ", methods_fg)
            sys.exit()  # TODO raise instead
        method_bg = args["--bg-method"]
        if method_bg not in methods_bg:
            print(__doc__)
            print("METHOD must be one of: ", methods_bg)
            sys.exit()  # TODO raise instead
        # bg() kwargs:
        kwargs_bg = {}
        kwargs_bg["kind"] = method_bg
        if args["--bg-downscale"]:
            downscale = args["--bg-downscale"].split(",")
            kwargs_bg["downscale"] = int(downscale[0]), int(downscale[1])
        if args["--bg-radius"]:
            kwargs_bg["radius"] = float(args["--bg-radius"])
        if args["--bg-adaptive-radius"]:
            kwargs_bg["adaptive_radius"] = float(args["--bg-adaptive-radius"])
        if args["--bg-percentile"]:
            kwargs_bg["perc"] = float(args["--bg-percentile"])
        if args["--bg-percentile-filter"]:
            kwargs_bg["arcsinh_perc"] = float(args["--bg-percentile-filter"])
        # d_mask_label() kwargs:
        kwargs_mask_label = {}
        kwargs_mask_label["channels"] = channels
        kwargs_mask_label["threshold_method"] = method_fg
        if args["--min-size"]:
            kwargs_mask_label["min_size"] = float(args["--min-size"])
        if args["--clear-border"]:
            kwargs_mask_label["clear_border"] = True
        if args["--wiener"]:
            kwargs_mask_label["wiener"] = True
        if args["--watershed"]:
            kwargs_mask_label["watershed"] = True
        if args["--randomwalk"]:
            kwargs_mask_label["randomwalk"] = True
        # d_meas_props() kwargs:
        kwargs_meas_props = {}
        kwargs_meas_props["channels"] = channels
        if args["--no-image-ratios"]:
            kwargs_meas_props["ratios_from_image"] = False
        if args["--ratio-median-radii"]:
            radii = args["--ratio-median-radii"].split(",")
            kwargs_meas_props["radii"] = tuple(int(r) for r in radii)
        if args["--channels-cl"]:
            kwargs_meas_props["channels_cl"] = args["--channels-cl"].split("/")
        if args["--channels-pH"]:
            kwargs_meas_props["channels_pH"] = args["--channels-pH"].split("/")
        print(kwargs_meas_props)

        # computation
        d_im, _, t = ni.read_tiff(args["TIFFSTK"], channels)
        if not args["--silent"]:
            print("  Times: ", t)
        if args["--hotpixels"]:
            d_im = ni.d_median(d_im)
        if args["--flat"]:
            dark, _, _ = ni.read_tiff(args["--dark"], channels)
            flat, _, _ = ni.read_tiff(args["--flat"], channels)
            d_im = ni.d_shading(d_im, dark, flat, clip=True)
        d_im_bg, bgs, ff, _bgv = ni.d_bg(d_im, **kwargs_bg)  # clip=True
        # dim I got a problem with 'li' and unique label for 19 1.10_15 af16 ds
        ni.d_mask_label(d_im_bg, **kwargs_mask_label)
        meas, pr = ni.d_meas_props(d_im_bg, **kwargs_meas_props)

        # output for bg
        bname = os.path.basename(args["TIFFSTK"])
        bname = os.path.splitext(bname)[0]
        # bname = os.path.join('nimg', bname)
        bname = os.path.join(args["--output"], bname)
        if not os.path.exists(bname):
            os.makedirs(bname)
        bname_bg = os.path.join(bname, "bg")
        for ch in ff.keys():
            pp = PdfPages(bname_bg + "-" + ch + "-" + method_bg + ".pdf")
            for t, f in enumerate(ff[ch]):
                for f_i in f:
                    pp.savefig(f_i)  # e.g. entropy output 2 figs
            pp.close()
        column_order = ["C", "G", "R"]
        bgs[column_order].to_csv(bname_bg + ".csv")
        # TODO: plt.close('all') or control mpl warning

        # output for fg (target)
        f = ni.d_plot_meas(bgs, meas, channels=channels)
        f.savefig(bname + "_meas.png")
        ##
        # show all channels and labels only.
        d = {ch: d_im_bg[ch] for ch in channels}
        d["labels"] = d_im_bg["labels"]
        f = ni.d_show(d, cmap=plt.cm.inferno_r)
        f.savefig(bname + "_dim.png")
        ##
        # meas csv
        for k, df in meas.items():
            column_order = [
                "C",
                "G",
                "R",
                "area",
                "eccentricity",
                "equivalent_diameter",
                "r_cl",
                "r_pH",
                "r_cl_median",
                "r_pH_median",
            ]
            df[column_order].to_csv(os.path.join(bname, "label" + str(k) + ".csv"))
        ##
        # labelX_{rcl,rpH}.tif ### require r_cl and r_pH present in d_im
        objs = ndimage.find_objects(d_im_bg["labels"])
        for k, o in enumerate(objs):
            name = os.path.join(bname, "label" + str(k + 1) + "_rcl.tif")
            tifffile.imsave(name, d_im_bg["r_cl"][o], compress=9)
            name = os.path.join(bname, "label" + str(k + 1) + "_rpH.tif")
            tifffile.imsave(name, d_im_bg["r_pH"][o], compress=9)


def dark(fp, thr=95):
    """
    Read zip; median z-project; median filter(1).
    Return imf. Plot imf and its histogram.
    thr: float threshold for hot pixels calculation.
    """
    im = zipread(fp)
    zp = ni.zproject(im)
    imf = ni.im_median(zp)
    f = plt.figure(figsize=(6.75, 9.25))
    plt.suptitle("DARK stack")
    #
    with plt.style.context("seaborn-ticks"):
        plt.subplot(321)
        plt.hist(imf.ravel(), bins=256, histtype="step", lw=4)
        plt.yscale("log")
        plt.title("DARK image")
    #
    plt.subplot(322)
    plt.imshow(imf, cmap=plt.cm.inferno_r)
    plt.colorbar()
    plt.axis("off")
    plt.title("exported DARK image")
    #
    with plt.style.context("seaborn-ticks"):
        plt.subplot(323)
        plt.hist(im.ravel(), bins=256, histtype="step", lw=4)
        plt.yscale("log")
        plt.title("original stack")
    #
    plt.subplot(324)
    # hot pixels; cast to float because uint screwed up to range max
    d = imf.astype(float) - zp.astype(float)
    thr = np.std(d) * thr
    hot_pixels = np.nonzero(abs(d) > thr)
    df_hp = pd.DataFrame(
        {"row": hot_pixels[0], "col": hot_pixels[1], "val": zp[hot_pixels]}
    )
    plt.imshow(zp)
    plt.plot(hot_pixels[1], hot_pixels[0], "r+", mfc="none", mec="w", ms=18)
    plt.colorbar()
    plt.axis("off")
    plt.title("projected stack")
    #
    plt.tight_layout()
    return imf, df_hp, f


def flat(fflat, fdark, method="overall"):
    """
    Return a flat image:
    dark subtracted and normalized either:
    at each plane 'single' or
    after median projection 'overall'
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
        flat = ni.im_median(flat)
        flat = flat / np.mean(flat)
    if method == "single":
        ims = ims.astype(float)
        for i, im in enumerate(ims):
            ims[i] = ni.im_median(im)
            ims[i] = ims[i] / np.mean(ims[i])
        flat = np.median(ims, axis=0)
    # Pdf output
    f = plt.figure(figsize=(6.75, 9.25))
    f.suptitle("FLAT stack")
    # table
    ax = plt.subplot2grid((6, 4), (0, 0), colspan=4)
    ax.set_axis_off()
    # http://nipunbatra.github.io/2014/08/latexify/
    params = {
        # 'axes.labelsize': 8,  # fontsize for x and y labels
        "font.size": 9,
        "font.family": "serif",
    }
    mpl.rcParams.update(params)
    fcommon, fdark_relative, fflat_relative = common_path(fdark, fflat)
    data = pd.Series(
        [fcommon, fdark_relative, fflat_relative],
        name="Files",
        index=["root", "dark", "flat"],
    )
    pd.tools.plotting.table(ax=ax, data=data, loc=3)
    mpl.rcdefaults()
    # FLAT
    ax0 = plt.subplot2grid((6, 4), (1, 0), colspan=3, rowspan=3)
    plt.axis("off")
    img0 = ax0.imshow(flat)
    plt.title("exported FLAT image")
    ax1 = plt.subplot2grid((6, 4), (1, 3), colspan=1, rowspan=3)
    plt.axis("off")
    plt.colorbar(img0, ax=ax1, fraction=0.9, shrink=0.78, aspect=14)
    #
    # hist flat
    with plt.style.context("seaborn-ticks"):
        plt.subplot2grid((6, 4), (4, 0), colspan=2, rowspan=2)
        plt.hist(flat.ravel(), bins=256, histtype="step", lw=2, color="crimson")
        plt.yscale("log")
        plt.grid()
        plt.title("FLAT image")
    # hist stack
    with plt.style.context("seaborn-ticks"):
        plt.subplot2grid((6, 4), (4, 2), colspan=2, rowspan=2)
        plt.hist(im.ravel(), bins=256, histtype="step", lw=2, color="grey")
        plt.yscale("log")
        plt.ylim(
            0.1,
        )
        plt.grid()
        plt.title("original stack")
    #
    plt.tight_layout()
    return flat, f


def common_path(path1, path2):
    """Utility function to find common absolute path.

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
    >>> common_path('/home/dan/docs/arte/unimi/grafEPS.tgz', \
                    '/home/dati/GBM_persson/')
    ('/home', 'dan/docs/arte/unimi/grafEPS.tgz', 'dati/GBM_persson')

    """
    fcommon = os.path.commonprefix([os.path.abspath(path1), os.path.abspath(path2)])
    fcommon = os.path.dirname(fcommon)  # stop at the directory in common path
    f1 = os.path.relpath(path1, start=fcommon)
    f2 = os.path.relpath(path2, start=fcommon)
    return fcommon, f1, f2


def zipread(fp):
    """ Unzip and read a single TIF. Return the image (np.array)."""
    with zipfile.ZipFile(fp) as myzip:
        with myzip.open(myzip.filelist[0]) as myfile:
            return io.imread(myfile)


if __name__ == "__main__":
    main()
