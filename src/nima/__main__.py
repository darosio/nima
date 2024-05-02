"""Command-line interface."""

import importlib.metadata
import os
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import click
import dask.array as da
import matplotlib as mpl
import numpy as np
import pandas as pd
import sigfig  # type: ignore[import-untyped]
import tifffile  # type: ignore[import-untyped]
from dask.diagnostics.progress import ProgressBar
from dask.distributed import Client, progress
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy import ndimage  # type: ignore[import-untyped]

from nima import nima
from nima.nima import Im, ImArray

from .segmentation import BgParams

__version__ = importlib.metadata.version("nima")
__out_dir__ = f"nima-{__version__}"


AXES_LENGTH_2D = 2


class _VerbosityLevel(int):
    SILENT = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


def ensure_ndarray(var: Any, var_name: str) -> None:  # noqa: ANN401
    """Ensure that the given variable is a numpy.ndarray."""
    if not isinstance(var, np.ndarray):
        msg = f"Expected '{var_name}' to be a numpy.ndarray"
        raise TypeError(msg)


@click.command()
@click.version_option(version=__version__, message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
@click.option("--silent", "-s", is_flag=True, help="Suppress output; verbose=0.")
@click.option("-o", "--output", default=__out_dir__, type=click.Path(writable=True, path_type=Path), # noqa: E501
              help="Output directory path [default: ./nima-version/].")  # fmt: skip
@click.option("--hotpixels", is_flag=True, default=False,
              help="Apply median filter (rad=0.5) to remove hot pixels.")  # fmt: skip
@click.option("-f", "--flat", "flat_f", type=str, default="",
              help="Path to flat image for shading correction.")  # fmt: skip
@click.option("-d", "--dark", "dark_f", type=str, default="",
              help="Path to dark image for shading correction.")  # fmt: skip
# Background estimation options
@click.option("--bg-method",
              type=click.Choice(["li_adaptive", "entropy", "arcsinh", "adaptive", "li_li"], case_sensitive=False),  # noqa: E501
              default="li_adaptive",
              help="Background estimation algorithm [default: li_adaptive].")  # fmt: skip # noqa: E501
@click.option("--bg-downscale", type=(int, int),
              help="Binning Y X.")  # fmt: skip
@click.option("--bg-radius", type=float,
              help="Radius for entropy or arcsinh methods [default: 10].")  # fmt: skip
@click.option("--bg-adaptive-radius", type=float,
              help="Radius for adaptive methods [default: X/2].")  # fmt: skip
@click.option("--bg-percentile", type=float,
              help="Percentile for entropy or arcsinh methods [default: 10].")  # fmt: skip # noqa: E501
@click.option("--bg-percentile-filter", type=float,
              help="Percentile filter for arcsinh method [default: 80].")  # fmt: skip
# Segmentation and measurement options
@click.option("--fg-method", type=click.Choice(["yen", "li"], case_sensitive=False), default="yen", # noqa: E501
              help="Segmentation algorithm [default: yen].")  # fmt: skip
@click.option("--min-size", type=float,
              help="Minimum size of labeled objects [default: 2000].")  # fmt: skip
@click.option("--clear-border", is_flag=True,
              help="Remove labels touching image borders [default: 0].")  # fmt: skip
@click.option("--wiener", is_flag=True,
              help="Apply Wiener filter before segmentation [default: 0].")  # fmt: skip
@click.option("--watershed", is_flag=True,
              help="Apply watershed binary mask (labeling) [default: 0].")  # fmt: skip
@click.option("--randomwalk", is_flag=True,
              help="Apply randomwalk binary mask (labeling) [default: 0].")  # fmt: skip
@click.option("--image-ratios/--no-image-ratios", default=True,
              help="Compute ratio images? [default: True].")  # fmt: skip
@click.option("--ratio-median-radii", type=str,
              help="Median filter ratio images with radii [default: (7, 3)].")  # fmt: skip # noqa: E501
@click.option("--channels-cl", type=(str, str), default=("C", "R"),
              help="Channels for Cl ratio [default: C/R].")  # fmt: skip
@click.option("--channels-ph", type=(str, str), default=("G", "C"),
              help="Channels for pH ratio [default: G/C].")  # fmt: skip
@click.argument("tiffstk", type=click.Path(path_type=Path))
@click.argument("channels", type=str, nargs=-1)
def main(  # noqa: PLR0913
    verbose: int,
    silent: bool | None,
    output: Path,
    hotpixels: bool,  # noqa: FBT001
    flat_f: str,
    dark_f: str,
    bg_method: str,
    bg_downscale: tuple[int, int] | None,
    bg_radius: float | None,
    bg_adaptive_radius: float | None,
    bg_percentile: float | None,
    bg_percentile_filter: float | None,
    fg_method: str,
    min_size: float | None,
    clear_border: bool | None,
    wiener: bool | None,
    watershed: bool | None,
    randomwalk: bool | None,
    image_ratios: bool,  # noqa: FBT001
    ratio_median_radii: str | None,
    channels_cl: tuple[str, str],
    channels_ph: tuple[str, str],
    tiffstk: Path,
    channels: tuple[str, ...],
) -> None:
    """Analyze a multichannel TIFF time-lapse stack.

    tiffstk : str
        Path to the TIFF image file.

    channels : list of str, optional
        Names of the channels in the TIFF image. Default is ["G", "R", "C"].

    Saves
    -----
    1. Representation of image channels and segmentation saved as `BN_dim.png`.
    2. Plot of ratios and channel intensities for each label and background vs.
       time saved as `BN_meas.png`.
    3. Table of background values saved as `f_name/bg.csv`.
    4. Representation of background image and histogram at all time points for
       each channel saved as `BN/bg-[C1,C2,⋯]-method.pdf`.
    5. For each label: Table of ratios and measured properties saved as
       `BN/label[1,2,⋯].csv`.
    6. For each label: Ratio images saved as `BN/label[1,2,⋯]_r[cl,pH].tif`.

    """
    verbose = 0 if silent else max(1, min(4, verbose))
    channels = ("G", "R", "C") if len(channels) == 0 else channels
    if verbose > _VerbosityLevel.SILENT:
        click.echo(tiffstk)
        click.echo(channels)
    d_im, _, t = nima.read_tiff(tiffstk, channels)
    if verbose > _VerbosityLevel.SILENT:
        click.echo(f"  Times: {t}")
    if hotpixels:
        d_im = nima.d_median(d_im)
    if flat_f:
        # XXX: this is imperfect: dark must be present of flat
        dark, _, _ = nima.read_tiff(Path(dark_f), channels)
        flat, _, _ = nima.read_tiff(Path(flat_f), channels)
        d_im = nima.d_shading(d_im, dark, flat, clip=True)
    # Process background
    kwargs_bg: dict[str, Any] = {"kind": bg_method}
    optional_keys = {
        "downscale": bg_downscale,
        "radius": bg_radius,
        "adaptive_radius": bg_adaptive_radius,
        "perc": bg_percentile,
        "arcsinh_perc": bg_percentile_filter,
    }
    kwargs_bg.update({key: value for key, value in optional_keys.items() if value})
    d_im_bg, bgs, ff, _bgv = nima.d_bg(
        d_im, BgParams(**kwargs_bg), downscale=bg_downscale
    )
    print(BgParams(**kwargs_bg))
    # Segment
    kwargs_mask_label: dict[str, Any] = {
        "channels": channels,
        "threshold_method": fg_method,
    }
    optional_keys = {
        "min_size": min_size,
        "clear_border": clear_border,
        "wiener": wiener,
        "watershed": watershed,
        "randomwalk": randomwalk,
    }
    kwargs_mask_label.update({k: v for k, v in optional_keys.items() if v})
    click.secho(kwargs_mask_label)
    nima.d_mask_label(d_im_bg, **kwargs_mask_label)
    # Measure
    kwargs_meas_props: dict[str, Any] = {"channels": channels}
    kwargs_meas_props["ratios_from_image"] = image_ratios
    if ratio_median_radii:
        kwargs_meas_props["radii"] = tuple(
            int(r) for r in ratio_median_radii.split(",")
        )
    click.secho(kwargs_meas_props)
    meas, pr = nima.d_meas_props(
        d_im_bg, channels_cl=channels_cl, channels_ph=channels_ph, **kwargs_meas_props
    )
    output_results(output, tiffstk, ff, meas, channels, d_im_bg, bg_method, bgs)


def output_results(  # noqa: PLR0913
    output_dir: Path,
    tiffstk: Path,
    ff: dict[str, list[list[Figure]]],
    meas: dict[np.int32, pd.DataFrame],
    channels: tuple[str, ...],
    d_im_bg: dict[str, Im],
    bg_method: str,
    bgs: pd.DataFrame,
) -> None:
    """Output results: csv tables and png images."""
    output_dir.mkdir(exist_ok=True)
    # Create file-named directory
    bname = output_dir / tiffstk.with_suffix("").name
    bname.mkdir(exist_ok=True)
    # Create PDF files
    for ch, llf in ff.items():
        pdf_file = bname / Path(f"bg-{ch}-{bg_method}.pdf")
        with PdfPages(pdf_file) as pp:  # type: ignore[no-untyped-call]
            for lf in llf:
                for f_i in lf:
                    pp.savefig(f_i)
    # Create CSV file
    column_order = ["C", "G", "R"]  # FIXME must be equal anyway in testing
    bgs[column_order].to_csv(bname / "bg.csv")
    # TODO: plt.close('all') or control mpl warning
    # Create measurement plots
    f = nima.d_plot_meas(bgs, meas, channels=channels)
    f.savefig(bname.with_name(bname.name + "_meas.png"))
    # Show all channels and labels
    d = {ch: d_im_bg[ch] for ch in channels}
    d["labels"] = d_im_bg["labels"]
    fig = nima.d_show(d, cmap=mpl.cm.inferno_r)  # type: ignore[attr-defined]
    fig.savefig(bname.with_name(bname.name + "_dim.png"))
    # Create measurement CSV files
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
        df[column_order].to_csv(bname / Path(f"label{k}.csv"))
    # XXX: labelX_{rcl,rpH}.tif ### require r_cl and r_pH present in d_im
    # Create TIFF files
    objs = ndimage.find_objects(d_im_bg["labels"])
    for n, o in enumerate(objs):
        name = bname / Path(f"label{n+1}_rcl.tif")
        tifffile.imwrite(
            name, d_im_bg["r_cl"][o], compression="lzma", photometric="minisblack"
        )
        name = bname / Path(f"label{n+1}_rpH.tif")
        tifffile.imwrite(
            name, d_im_bg["r_pH"][o], compression="lzma", photometric="minisblack"
        )


##  bima  ##################################################
@click.group()
@click.pass_context
@click.version_option()
@click.option("-o", "--output", type=click.Path(writable=True, path_type=Path),
              help="Output path [default: *.tif, *.png].")  # fmt: skip
def bima(ctx: click.Context, output: Path) -> None:
    """Compute bias, dark and flat."""
    ctx.ensure_object(dict)
    ctx.obj["output"] = output


@bima.command()
@click.pass_context
@click.argument("fpath", type=click.Path(path_type=Path))
def bias(ctx: click.Context, fpath: Path) -> None:
    """Compute the BIAS frame and estimate read noise.

    fpath : str
        Path to the bias stack (Light Off - 0 acquisition time).

    Saves
    -----
    1. BIAS image (.tif): Median projection.
    2. Plot (.png): Includes histograms, median projection, and visualization of
       hot pixels.
    3. Hot pixel coordinates and values (.csv): If hot pixels are detected.

    """
    if fpath.suffix == ".zip":
        with zipfile.ZipFile(fpath) as zf, zf.open(zf.namelist()[0]) as fo:
            # Alternatively: ignore store = tifffile.imread(fo)
            store = tifffile.imread(BytesIO(fo.read()))
    else:
        store = tifffile.imread(fpath)
    ensure_ndarray(store, "store")
    click.secho("Bias image-stack shape: " + str(store.shape), fg="green")
    bias = np.median(store, axis=0)
    err = np.std(store, axis=0)
    # hotpixels
    hpix = nima.hotpixels(bias)
    output = ctx.obj["output"] if ctx.obj["output"] else fpath.with_suffix(".png")
    if not hpix.empty:
        hpix.to_csv(output.with_suffix(".csv"), index=False)
        # FIXME hpix.y is a pd.Series[int]; it could be cast into NDArray[int]
        # TODO: if any of x y is out of the border ignore them
        nima.correct_hotpixel(err, hpix.y, hpix.x)  # type: ignore[arg-type]
    p25, p50, p75 = np.percentile(err.ravel(), [25, 50, 75])
    err_str = sigfig.round(p50, p75 - p25)
    click.secho("Estimated read noise: " + err_str)
    tifffile.imwrite(output.with_suffix(".tiff"), bias)
    # Output summary graphics.
    title = os.fspath(output.with_suffix("").name)
    if bias.ndim == AXES_LENGTH_2D:
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
@click.option("--bias", type=click.Path(path_type=Path),
              help="File path to the bias stack (Light Off - Long acquisition time).")  # fmt: skip # noqa: E501
@click.option("--time", type=float,
              help="Acquisition time.")  # fmt: skip
@click.argument("fpath", type=click.Path(path_type=Path))
def dark(ctx: click.Context, fpath: Path, bias: Path, time: float) -> None:
    """Compute DARK.

    fpath : str
        Path to the dark stack (Light Off - Long acquisition time).

    Saves
    -----
    1. DARK image (.tif): Median projection.
    2. Plot (.png): Includes histograms, median projection, ...

    """
    dark_thr = 4.5
    store = tifffile.imread(fpath)
    ensure_ndarray(store, "store")
    click.secho("Dark image-stack shape: " + str(store.shape), fg="green")
    dark = np.median(store, axis=0)
    output = ctx.obj["output"] if ctx.obj["output"] else fpath.with_suffix(".png")
    # Output summary graphics.
    title = os.fspath(output.with_suffix("").name)
    if bias is not None:
        bias_frame = np.array(tifffile.imread(bias))
        dark = dark - bias_frame
    if time:
        dark /= time
    plt_img_profiles(dark, title, output)
    print(np.where(dark > dark_thr))


@bima.command()
@click.pass_context
@click.option("--bias", type=click.Path(path_type=Path),
              help="Path to the bias stack (Light Off - 0 acquisition time).")  # fmt: skip # noqa: E501
@click.argument("globpath", type=str)
def mflat(ctx: click.Context, globpath: str, bias: Path | None) -> None:
    """Compute the flat field from a collection of (.tif) files.

    globpath : "glob expression"
        Glob pattern (enclosed in quotes) for a collection of (.tif) files.

    Saves
    -----
    1. FLAT image (.tif): Mean projection.
    2. Plot (.png): Includes histograms, mean projection, ...

    """
    image_sequence = tifffile.TiffSequence(globpath)
    axes_n_shape = " ".join((str(image_sequence.axes), str(image_sequence.shape)))
    click.secho(axes_n_shape, fg="green")
    store = image_sequence.aszarr()
    Client()  # type: ignore[no-untyped-call, unused-awaitable]
    f = da.mean(da.from_zarr(store).rechunk(), axis=0)  # type: ignore[attr-defined, no-untyped-call]
    fp = f.persist()
    progress(fp)  # type: ignore[no-untyped-call]
    tprojection = fp.compute()
    if ctx.obj["output"]:
        output = ctx.obj["output"]
    else:
        output = Path(globpath).name.replace("*", "").replace("?", "")
        output = Path(output).with_suffix(".tiff")
    bias_frame = None if bias is None else np.array(tifffile.imread(bias))
    _output_flat(output, tprojection, bias_frame)


@bima.command()
@click.pass_context
@click.option("--bias", type=click.Path(path_type=Path),
              help="Path to the bias stack (Light Off - 0 acquisition time).")  # fmt: skip # noqa: E501
@click.argument("fpath", type=click.Path(path_type=Path))
def flat(ctx: click.Context, fpath: Path, bias: Path) -> None:
    """Flat from (.tf8) file stack.

    fpath : str
        Path to the (.tf8) file containing the image data.

    Saves
    -----
    1. FLAT image (.tif): Mean projection.
    2. Plot (.png): Includes histograms, mean projection, ...

    """
    store = tifffile.imread(fpath, aszarr=True)
    f = da.mean(da.from_zarr(store).rechunk(), axis=0)  # type: ignore[attr-defined, no-untyped-call]
    with ProgressBar():  # type: ignore[no-untyped-call]
        tprojection = f.compute()
    output = ctx.obj["output"] if ctx.obj["output"] else fpath.with_suffix(".tiff")
    bias_frame = np.array(tifffile.imread(bias))
    _output_flat(output, tprojection, bias_frame)


def _output_flat(
    output: Path, tprojection: ImArray, bias: ImArray | None = None
) -> None:
    """Help to generate and save output files from flat field calculations.

    The function performs the following tasks:
    - Saves the raw mean of frames to a file with a '_raw.tif' suffix.
    - If a bias frame is provided, it subtracts this from the raw mean,
      smooths the result using a Gaussian filter, and normalizes the smoothed
      image. This is saved to a '.tif' file.
    - Generates summary graphics and saves as '.png'.

    Parameters
    ----------
    output : Path
        Base path for generating output file names.
    tprojection : ImArray
        2D array representing the raw flat field image (mean of frames).
    bias : ImArray | None
        2D array representing the bias frame for subtraction.
        If None (default), no bias subtraction is performed.

    Notes
    -----
    The constant value (e.g., 20) added to 'tprojection' before subtracting
    'bias' in the function's implementation may need further review or
    adjustment based on the specific requirements of the flat field correction.

    """
    tifffile.imwrite(output.with_stem("-".join([output.stem, "raw"])), tprojection)
    if bias is None:
        flat = ndimage.gaussian_filter(tprojection, sigma=100)
    else:
        flat = ndimage.gaussian_filter(tprojection + 20 - bias, sigma=100)  # FIXME
        # MAYBE: consider skimage.filters.gaussian and  cmap=plt.cm.Set2_r
    flat /= flat.mean()
    tifffile.imwrite(output, flat)
    title = os.fspath(output.with_suffix("").name)
    plt_img_profiles(flat, title, output)


@bima.command()
@click.pass_context
@click.argument("fpath", type=click.Path(exists=True, path_type=Path))
def plot(ctx: click.Context, fpath: Path) -> None:
    """Plot profiles of a 2D image.

    fpath : str
        Path to the 2D image file (e.g. Bias or Dark).

    Saves
    -----
    A plot of profiles is saved as a '.png' file.

    """
    img = np.array(tifffile.imread(fpath))
    output = ctx.obj["output"] if ctx.obj["output"] else fpath.with_suffix(".png")
    title = os.fspath(output.with_suffix("").name)
    plt_img_profiles(img, title, output)


def plt_img_profiles(
    img: ImArray, title: str, output: Path, hpix: pd.DataFrame | None = None
) -> None:
    """Compute and save image profiles graphics."""
    if img.ndim == AXES_LENGTH_2D:
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
