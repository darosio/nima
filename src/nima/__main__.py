"""Command-line interface."""
from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import click
import dask.array as da
import matplotlib as mpl
import numpy as np
import pandas as pd
import sigfig  # type: ignore
import tifffile  # type: ignore
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client, progress
from matplotlib.backends import backend_pdf  # type: ignore
from scipy import ndimage  # type: ignore

from nima import nima
from nima.nima import ImArray


@click.command()
@click.version_option()
@click.option("--silent", is_flag=True, help="Do not print; verbose=0.")
@click.option(
    "-o",
    "--output",
    default="nima",
    type=click.Path(writable=True, path_type=Path),
    help="Output path [default: ./nima/].",
)
@click.option(
    "--hotpixels",
    is_flag=True,
    default=False,
    help="Median filter (rad=0.5) to remove hotpixels.",
)
@click.option("-f", "--flat", type=str, help="Dark for shading correction.")
@click.option("-d", "--dark", type=str, help="Flat for shading correction.")
@click.option(
    "--bg-method",
    type=click.Choice(
        ["li_adaptive", "entropy", "arcsinh", "adaptive", "li_li"], case_sensitive=False
    ),
    default="li_adaptive",
    prompt_required=False,
    prompt=True,
    help="Background estimation algorithm [default:li_adaptive].",
)
@click.option("--bg-downscale", type=(int, int), help="Binning Y X.")
@click.option(
    "--bg-radius", type=float, help="Radius for entropy or arcsinh methods [def:10]."
)
@click.option(
    "--bg-adaptive-radius", type=float, help="Radius for adaptive methods [def:X/2]."
)
@click.option(
    "--bg-percentile",
    type=float,
    help="Percentile for entropy or arcsinh methods [def:10].",
)
@click.option(
    "--bg-percentile-filter",
    type=float,
    help="Percentile filter for arcsinh method [def:80].",
)
@click.option(
    "--fg-method",
    type=click.Choice(["yen", "li"], case_sensitive=False),
    default="yen",
    prompt_required=False,
    prompt=True,
    help="Segmentation algorithm [default:yen].",
)
@click.option("--min-size", type=float, help="Minimum size labeled objects [def:2000].")
@click.option(
    "--clear-border", is_flag=True, help="Remove labels touching image borders."
)
@click.option("--wiener", is_flag=True, help="Wiener filter before segmentation.")
@click.option(
    "--watershed", is_flag=True, help="Watershed binary mask (to label cells)."
)
@click.option(
    "--randomwalk", is_flag=True, help="Randomwalk binary mask (to label cells)."
)
#
@click.option(
    "--image-ratios/--no-image-ratios",
    default=True,
    help="Compute ratio images? [default:True]",
)
@click.option(
    "--ratio-median-radii",
    type=str,
    help="Median filter ratio images with radii [def:(7,3)].",
)
@click.option(
    "--channels-cl",
    type=(str, str),
    default=("C", "R"),
    help="Channels for Cl ratio [default:C/R].",
)
@click.option(
    "--channels-ph",
    type=(str, str),
    default=("G", "C"),
    help="Channels for pH ratio [default:G/C].",
)
# # TODO: @click.argument("tiffstk", type=click.File("r"))
@click.argument("tiffstk", type=click.Path(path_type=Path))
# @click.argument("channels", type=list[str], default=["G", "R", "C"])
@click.argument("channels", type=str, nargs=-1)
def main(  # type: ignore # noqa: C901"
    silent,
    output,
    hotpixels,
    flat,
    dark,
    fg_method,
    min_size,
    clear_border,
    wiener,
    watershed,
    randomwalk,
    bg_method,
    bg_downscale,
    bg_radius,
    bg_adaptive_radius,
    bg_percentile,
    bg_percentile_filter,
    image_ratios,
    ratio_median_radii,
    channels_cl,
    channels_ph,
    tiffstk,
    channels,
):
    """Analyze multichannel (default:["G", "R", "C"]) tiff time-lapse stack.

    TIFFSTK  :  Image file.

    CHANNELS :  Channel names.

    Save:
    (1) representation of image channels and segmentation ``BN_dim.png``,
    (2) plot of ratios and channel intensities for each label and bg vs.
    time ``BN_meas.png``,
    (3) table of bg values ``*/bg.csv``,
    (4) representation of bg image and histogram at all time points for
    each channel ``BN/bg-[C1,C2,⋯]-method.pdf``, and for each label:
    (5) table of ratios and measured properties ``BN/label[1,2,⋯].csv``
    and (6) ratio images ``BN/label[1,2,⋯]_r[cl,pH].tif``.
    """
    click.echo(tiffstk)
    channels = ("G", "R", "C") if len(channels) == 0 else channels
    click.echo(channels)
    d_im, _, t = nima.read_tiff(tiffstk, channels)
    if not silent:
        print("  Times: ", t)
    if hotpixels:
        d_im = nima.d_median(d_im)
    if flat:
        # XXX: this is imperfect: dark must be present of flat
        dark, _, _ = nima.read_tiff(dark, channels)
        flat, _, _ = nima.read_tiff(flat, channels)
        d_im = nima.d_shading(d_im, dark, flat, clip=True)
    kwargs_bg = {"kind": bg_method}
    if bg_downscale:
        kwargs_bg["downscale"] = bg_downscale
    if bg_radius:
        kwargs_bg["radius"] = bg_radius
    if bg_adaptive_radius:
        kwargs_bg["adaptive_radius"] = bg_adaptive_radius
    if bg_percentile:
        kwargs_bg["perc"] = bg_percentile
    if bg_percentile_filter:
        kwargs_bg["arcsinh_perc"] = bg_percentile_filter
    click.echo(kwargs_bg)
    d_im_bg, bgs, ff, _bgv = nima.d_bg(d_im, **kwargs_bg)  # clip=True
    print(bgs)

    kwargs_mask_label = {"channels": channels, "threshold_method": fg_method}
    if min_size:
        kwargs_mask_label["min_size"] = min_size
    if clear_border:
        kwargs_mask_label["clear_border"] = True
    if wiener:
        kwargs_mask_label["wiener"] = True
    if watershed:
        kwargs_mask_label["watershed"] = True
    if randomwalk:
        kwargs_mask_label["randomwalk"] = True
    click.secho(kwargs_mask_label)
    nima.d_mask_label(d_im_bg, **kwargs_mask_label)
    kwargs_meas_props = {"channels": channels}
    kwargs_meas_props["ratios_from_image"] = image_ratios
    if ratio_median_radii:
        kwargs_meas_props["radii"] = tuple(
            int(r) for r in ratio_median_radii.split(",")
        )
    click.secho(kwargs_meas_props)
    meas, pr = nima.d_meas_props(
        d_im_bg, channels_cl=channels_cl, channels_ph=channels_ph, **kwargs_meas_props
    )
    #     # output for bg
    bname = tiffstk.with_suffix("").name
    output.mkdir(exist_ok=True)
    bname = output / bname
    if not bname.exists():
        bname.mkdir()
    for ch, llf in ff.items():
        pp = backend_pdf.PdfPages(
            bname / Path("bg-" + ch + "-" + bg_method).with_suffix(".pdf")
        )
        for lf in llf:
            for f_i in lf:
                pp.savefig(f_i)  # e.g. entropy output 2 figs
        pp.close()
    column_order = ["C", "G", "R"]  # FIXME must be equal anyway in testing
    bgs[column_order].to_csv(bname / "bg.csv")
    # TODO: plt.close('all') or control mpl warning
    # output for fg (target)
    f = nima.d_plot_meas(bgs, meas, channels=channels)
    f.savefig(bname.with_name(bname.name + "_meas.png"))
    ##
    # show all channels and labels only.
    d = {ch: d_im_bg[ch] for ch in channels}
    d["labels"] = d_im_bg["labels"]
    f = nima.d_show(d, cmap=mpl.cm.inferno_r)  # type: ignore
    f.savefig(bname.with_name(bname.name + "_dim.png"))
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
        df[column_order].to_csv(bname / Path("label" + str(k)).with_suffix(".csv"))
    # ##
    # labelX_{rcl,rpH}.tif ### require r_cl and r_pH present in d_im
    objs = ndimage.find_objects(d_im_bg["labels"])
    for n, o in enumerate(objs):
        name = bname / Path("label" + str(n + 1) + "_rcl").with_suffix(".tif")
        tifffile.imwrite(
            name, d_im_bg["r_cl"][o], compression="lzma", photometric="minisblack"
        )
        name = bname / Path("label" + str(n + 1) + "_rpH").with_suffix(".tif")
        tifffile.imwrite(
            name, d_im_bg["r_pH"][o], compression="lzma", photometric="minisblack"
        )


#######################################################################################
@click.group()
@click.pass_context
@click.version_option()
@click.option(
    "-o",
    "--output",
    type=click.Path(writable=True, path_type=Path),
    help="Output path [default: *.tif, *.png].",
)
def bima(ctx: click.Context, output: Path) -> None:
    """Compute bias, dark and flat."""
    ctx.ensure_object(dict)
    ctx.obj["output"] = output


@bima.command()
@click.pass_context
@click.argument("fpath", type=click.Path(path_type=Path))
def bias(ctx: click.Context, fpath: Path) -> None:
    """Compute BIAS frame and estimate read noise.

    FPATH: the bias stack (Light Off - 0 acquisition time).

    Output:

    * .tif BIAS image = median projection

    * .png plot (histograms, median, projection, hot pixels)

    * [.csv coordinates and values of hot pixels] if detected

    """
    if fpath.suffix == ".zip":
        zf = zipfile.ZipFile(fpath)
        fo = zf.open(zf.namelist()[0])
        store = tifffile.imread(fo)
    else:
        store = tifffile.imread(fpath)
    click.secho("Bias image-stack shape: " + str(store.shape), fg="green")
    bias = np.median(store, axis=0)
    err = np.std(store, axis=0)
    # hotpixels
    hpix = nima.hotpixels(bias)
    if ctx.obj["output"]:
        output = ctx.obj["output"]
    else:
        output = fpath.with_suffix(".png")
    if not hpix.empty:
        hpix.to_csv(output.with_suffix(".csv"), index=False)
        # FIXME hpix.y is a pd.Series[int]; it could be cast into NDArray[int]
        # TODO: if any of x y is out of the border ignore them
        nima.correct_hotpixel(err, hpix.y, hpix.x)  # type: ignore
    p25, p50, p75 = np.percentile(err.ravel(), [25, 50, 75])
    err_str = sigfig.round(p50, p75 - p25)
    click.secho("Estimated read noise: " + err_str)
    tifffile.imwrite(output.with_suffix(".tiff"), bias)
    # Output summary graphics.
    title = os.fspath(output.with_suffix("").name)
    if bias.ndim == 2:
        plt_img_profiles(bias, title, output, hpix)
        plt_img_profiles(
            err,
            "".join(("[", title[:9], "] $\\sigma_{read} = $", err_str)),
            output.with_suffix(".err.png"),
        )
    else:
        for i in range(bias.shape[0]):
            plt_img_profiles(bias[i], title, output.with_suffix(f".{i}.png"), hpix)


@bima.command()
@click.pass_context
@click.option("--bias", type=click.Path(path_type=Path))
@click.option("--time", type=float)
@click.argument("fpath", type=click.Path(path_type=Path))
def dark(ctx: click.Context, fpath: Path, bias: Path, time: float) -> None:
    """Compute DARK.

    FPATH: the bias stack (Light Off - Long acquisition time).


    """
    store = tifffile.imread(fpath)
    click.secho("Dark image-stack shape: " + str(store.shape), fg="green")
    dark = np.median(store, axis=0)
    if ctx.obj["output"]:
        output = ctx.obj["output"]
    else:
        output = fpath.with_suffix(".png")
    # Output summary graphics.
    title = os.fspath(output.with_suffix("").name)
    if bias is not None:
        bias = tifffile.imread(bias)
        dark = dark - bias
    if time:
        dark /= time
    plt_img_profiles(dark, title, output)
    print(np.where(dark > 4.5))


@bima.command()
@click.pass_context
@click.option("--bias", type=click.Path(path_type=Path))
@click.argument("globpath", type=str)
def mflat(ctx: click.Context, globpath: str, bias: Path) -> None:
    """Flat from a collection of (.tif) files."""
    image_sequence = tifffile.TiffSequence(globpath)
    axes_n_shape = " ".join((str(image_sequence.axes), str(image_sequence.shape)))
    click.secho(axes_n_shape, fg="green")
    store = image_sequence.aszarr()
    Client()  # type: ignore
    f = da.mean(da.from_zarr(store).rechunk(), axis=0)  # type: ignore
    fp = f.persist()
    progress(fp)  # type: ignore
    tprojection = fp.compute()
    if ctx.obj["output"]:
        output = ctx.obj["output"]
    else:
        output = Path(globpath).name.replace("*", "").replace("?", "")
        output = Path(output).with_suffix(".tiff")
    if bias is not None:
        bias = tifffile.imread(bias)
    _output_flat(output, tprojection, bias)


@bima.command()
@click.pass_context
@click.option("--bias", type=click.Path(path_type=Path))
@click.argument("fpath", type=click.Path(path_type=Path))
def flat(ctx: click.Context, fpath: Path, bias: Path) -> None:
    """Flat from (.tf8) file."""
    store = tifffile.imread(fpath, aszarr=True)
    f = da.mean(da.from_zarr(store).rechunk(), axis=0)  # type: ignore
    with ProgressBar():  # type: ignore
        tprojection = f.compute()
    if ctx.obj["output"]:
        output = ctx.obj["output"]
    else:
        output = fpath.with_suffix(".tiff")
    if bias is not None:
        bias = tifffile.imread(bias)
    _output_flat(output, tprojection, bias)


def _output_flat(
    output: Path, tprojection: ImArray, bias: ImArray | None = None
) -> None:
    """Help output from flat calculations.

    - raw mean (_raw.tif)
    - bias subtracted, smoothed and normalized mean (.tif)
    - a summary graphics (.png)

    Parameters
    ----------
    output : Path
        Base for output file paths.
    tprojection : ImArray
        Raw flat array (mean of frames).
    bias : ImArray
        Bias frame.

    """
    if sys.version_info >= (3, 9):
        tifffile.imwrite(output.with_stem("-".join([output.stem, "raw"])), tprojection)
    else:
        rename = "-".join([output.stem, "raw"]) + output.suffix
        tifffile.imwrite(output.with_name(rename), tprojection)
    if bias is None:
        flat = ndimage.gaussian_filter(tprojection, sigma=100)
    else:
        flat = ndimage.gaussian_filter(tprojection + 20 - bias, sigma=100)  # FIXME
    flat /= flat.mean()
    tifffile.imwrite(output, flat)
    title = os.fspath(output.with_suffix("").name)
    plt_img_profiles(flat, title, output)


@bima.command()
@click.pass_context
@click.argument("fpath", type=click.Path(exists=True, path_type=Path))
def plot(ctx: click.Context, fpath: Path) -> None:
    """Plot profiles of 2D (Bias-Flat) image."""
    img = tifffile.imread(fpath)
    if ctx.obj["output"]:
        output = ctx.obj["output"]
    else:
        output = fpath.with_suffix(".png")
    title = os.fspath(output.with_suffix("").name)
    plt_img_profiles(img, title, output)


def plt_img_profiles(
    img: ImArray, title: str, output: Path, hpix: pd.DataFrame | None = None
) -> None:
    """Compute and save image profiles graphics."""
    if img.ndim == 2:
        f = nima.plt_img_profile(img, title=title, hpix=hpix)
        f.savefig(output.with_suffix(".png"), dpi=250, facecolor="w")
        # mark f = nima.plt_img_profile_2(img, title=title)
        # mark f.savefig(output.with_suffix(".2.png"), dpi=250, facecolor="w")
    else:
        for i in range(img.shape[0]):
            title += f" C:{i}"
            f = nima.plt_img_profile(img[i], title=title)
            f.savefig(output.with_suffix(f".C{i}.png"), dpi=250, facecolor="w")
            f = nima.plt_img_profile_2(img[i], title=title)
            f.savefig(output.with_suffix(f".C{i}.2.png"), dpi=250, facecolor="w")


if __name__ == "__main__":
    main(prog_name="nima")  # pragma: no cover
