"""Command-line interface."""

import importlib.metadata
import os
import zipfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from typing import Any

import click
import dask
import dask.array as da
import numpy as np
import pandas as pd
import sigfig  # type: ignore[import-untyped]
import tifffile
import xarray as xr
from dask.diagnostics.progress import ProgressBar
from matplotlib import cm, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from scipy import ndimage

from nima import io, nima

from .nima_types import ImFrame
from .segmentation import BgParams

__version__ = importlib.metadata.version("nima")
__out_dir__ = f"nima-{__version__}"
PATH_TYPE = click.Path(path_type=Path)  # type: ignore[type-var]
PATH_OUT = click.Path(path_type=Path, writable=True)  # type: ignore[type-var]
PATH_IN = click.Path(path_type=Path, exists=True)  # type: ignore[type-var]
AXES_LENGTH_2D = 2
AXES_LENGTH_3D = 3


def _compute_bias_hpix(
    bias_im: xr.DataArray, err: xr.DataArray
) -> tuple[xr.DataArray, pd.DataFrame]:
    if bias_im.ndim == AXES_LENGTH_2D:
        hpix = nima.hotpixels(bias_im)
        if not hpix.empty:
            err = nima.correct_hotpixel(err, hpix.y, hpix.x)  # type: ignore[arg-type]
        return err, hpix
    if bias_im.ndim == AXES_LENGTH_3D:
        hpix_dfs = []
        err_list = []
        for i in range(bias_im.shape[0]):
            b_ch = bias_im[i]
            e_ch = err[i]
            hp = nima.hotpixels(b_ch)
            if not hp.empty:
                hp["C"] = i
                hpix_dfs.append(hp)
                e_ch = nima.correct_hotpixel(e_ch, hp.y, hp.x)  # type: ignore[arg-type]
            err_list.append(e_ch)
        err = xr.concat(err_list, dim=bias_im.dims[0])
        hpix = pd.concat(hpix_dfs) if hpix_dfs else pd.DataFrame()
        return err, hpix
    return err, pd.DataFrame()


def _check_no_overwrite(inputs: Iterable[Path], outputs: Iterable[Path]) -> None:
    """Raise if any output path resolves to one of the input paths.

    Parameters
    ----------
    inputs : Iterable[Path]
        Paths of the input files.
    outputs : Iterable[Path]
        Paths the command is going to write.

    Raises
    ------
    click.BadParameter
        If an output would overwrite an input.
    """
    clashes = {p.resolve() for p in inputs} & {p.resolve() for p in outputs}
    if clashes:
        msg = f"Output would overwrite input {min(clashes)}; set -o/--output."
        raise click.BadParameter(msg)


def _plot_bias(
    bias_im: xr.DataArray,
    err: xr.DataArray,
    output: Path,
    hpix: pd.DataFrame,
    err_str: str,
) -> None:
    title = os.fspath(output.with_suffix("").name)
    if bias_im.ndim == AXES_LENGTH_2D:
        plt_img_profiles(bias_im, title, output, hpix)
        plt_img_profiles(
            err,
            "".join(("[", title[:9], "] $\\sigma_{read} = $", err_str)),
            output.with_suffix(".err.png"),
        )
    else:
        for i in range(bias_im.shape[0]):
            plt_img_profiles(bias_im[i], title, output.with_suffix(f".{i}.png"), hpix)


# MAYBE: Remove docstring and silent pydoclint
class _VerbosityLevel(int):
    """Manage verbosity level.

    Attributes
    ----------
    SILENT :
        Silent level (0).
    LOW :
        Low verbosity level (1).
    MEDIUM :
        Medium verbosity level (2).
    HIGH :
        High verbosity level (3).

    """

    SILENT = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3


def _drop_unset(options: dict[str, Any]) -> dict[str, Any]:
    """Return the options that were set, i.e. not None (zeros are kept).

    Parameters
    ----------
    options : dict[str, Any]
        Option names mapped to their command-line values.

    Returns
    -------
    dict[str, Any]
        The options whose value is not None.
    """
    return {k: v for k, v in options.items() if v is not None}


def _check_shading_pair(flat_f: Path | None, dark_f: Path | None) -> None:
    """Require flat and dark images together for shading correction.

    Parameters
    ----------
    flat_f : Path | None
        Flat image path.
    dark_f : Path | None
        Dark image path.

    Raises
    ------
    click.UsageError
        If only one of the two is given.
    """
    if (flat_f is None) != (dark_f is None):
        msg = "Shading correction needs both -f/--flat and -d/--dark."
        raise click.UsageError(msg)


def _parse_radii(
    _ctx: click.Context, _param: click.Parameter, value: str | None
) -> tuple[int, ...] | None:
    """Parse comma-separated positive integer radii, e.g. ``"7,3"``.

    Parameters
    ----------
    _ctx : click.Context
        Click context (unused).
    _param : click.Parameter
        Click parameter (unused).
    value : str | None
        Raw option value.

    Returns
    -------
    tuple[int, ...] | None
        The parsed radii, or None when the option is not given.

    Raises
    ------
    click.BadParameter
        If any radius is not a positive integer.
    """
    if value is None:
        return None
    try:
        radii = tuple(int(r) for r in value.split(","))
    except ValueError:
        radii = ()
    if not radii or min(radii) < 1:
        msg = f"expected comma-separated positive integers, got {value!r}"
        raise click.BadParameter(msg)
    return radii


@click.command()
@click.version_option(version=__version__, message="%(version)s")
@click.option("--verbose", "-v", count=True, help="Verbosity of messages.")
@click.option("--silent", "-s", is_flag=True, help="Suppress output; verbose=0.")
@click.option("-o", "--output", default=__out_dir__, type=PATH_OUT,
              help=f"Output directory path [default: ./{__version__}/].")  # fmt: skip
@click.option("--hotpixels", is_flag=True, default=False,
              help="Apply median filter (rad=0.5) to remove hot pixels.")  # fmt: skip
@click.option("-f", "--flat", "flat_f", type=PATH_IN, default=None,
              help="Path to flat image for shading correction.")  # fmt: skip
@click.option("-d", "--dark", "dark_f", type=PATH_IN, default=None,
              help="Path to dark image for shading correction.")  # fmt: skip
# Background estimation options
@click.option("--bg-method",
              type=click.Choice(["li_adaptive", "entropy", "arcsinh", "adaptive", "li_li"], case_sensitive=False),  # noqa: E501
              default="li_adaptive",
              help="Background estimation algorithm [default: li_adaptive].")  # fmt: skip # noqa: E501
@click.option("--bg-downscale", type=(click.IntRange(min=1), click.IntRange(min=1)),
              help="Binning Y X.")  # fmt: skip
@click.option("--bg-radius", type=click.IntRange(min=1),
              help="Radius for entropy or arcsinh methods [default: 10].")  # fmt: skip
@click.option("--bg-adaptive-radius", type=click.IntRange(min=1),
              help="Radius for adaptive methods [default: X/2].")  # fmt: skip
@click.option("--bg-percentile", type=click.FloatRange(0, 100),
              help="Percentile for entropy or arcsinh methods [default: 10].")  # fmt: skip # noqa: E501
@click.option("--bg-percentile-filter", type=click.FloatRange(0, 100),
              help="Percentile filter for arcsinh method [default: 80].")  # fmt: skip
# Segmentation and measurement options
@click.option("--fg-method", type=click.Choice(["yen", "li"], case_sensitive=False), default="yen",  # noqa: E501
              help="Segmentation algorithm [default: yen].")  # fmt: skip
@click.option("--min-size", type=click.IntRange(min=1),
              help="Minimum size of labeled objects [default: 640].")  # fmt: skip
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
@click.option("--ratio-median-radii", type=str, callback=_parse_radii,
              help="Median filter ratio images with radii [default: (7, 3)].")  # fmt: skip # noqa: E501
@click.option("--channels-cl", type=(str, str), default=("C", "R"),
              help="Channels for Cl ratio [default: C/R].")  # fmt: skip
@click.option("--channels-ph", type=(str, str), default=("G", "C"),
              help="Channels for pH ratio [default: G/C].")  # fmt: skip
@click.argument("tiffstk", type=PATH_IN)
@click.argument("channels", type=str, nargs=-1)
def main(  # noqa: PLR0913
    verbose: int,
    silent: bool | None,  # noqa: FBT001
    output: Path,
    hotpixels: bool,  # noqa: FBT001
    flat_f: Path | None,
    dark_f: Path | None,
    bg_method: str,
    bg_downscale: tuple[int, int] | None,
    bg_radius: int | None,
    bg_adaptive_radius: int | None,
    bg_percentile: float | None,
    bg_percentile_filter: float | None,
    fg_method: str,
    min_size: int | None,
    clear_border: bool | None,  # noqa: FBT001
    wiener: bool | None,  # noqa: FBT001
    watershed: bool | None,  # noqa: FBT001
    randomwalk: bool | None,  # noqa: FBT001
    image_ratios: bool,  # noqa: FBT001
    ratio_median_radii: tuple[int, ...] | None,
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

    Notes
    -----
    Saves:
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
    _check_shading_pair(flat_f, dark_f)
    verbose = 0 if silent else max(1, min(4, verbose))
    channels = ("G", "R", "C") if len(channels) == 0 else channels
    if verbose > _VerbosityLevel.SILENT:
        click.echo(tiffstk)
        click.echo(channels)
    im = io.read_image(tiffstk, channels)
    t = int(im.sizes["T"])
    if verbose > _VerbosityLevel.SILENT:
        click.echo(f"  Times: {t}")
    if hotpixels:
        im = nima.median(im)
    if flat_f and dark_f:
        dark_im = io.read_image(Path(dark_f), channels)
        flat_im = io.read_image(Path(flat_f), channels)
        im = nima.shading(im, dark_im, flat_im, clip=True)

    # Process background
    kwargs_bg: dict[str, Any] = {"kind": bg_method}
    optional_keys = {
        "radius": bg_radius,
        "adaptive_radius": bg_adaptive_radius,
        "perc": bg_percentile,
        "arcsinh_perc": bg_percentile_filter,
    }
    kwargs_bg.update(_drop_unset(optional_keys))
    im, bgs, ff = nima.bg(im, BgParams(**kwargs_bg), downscale=bg_downscale)

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
    kwargs_mask_label.update(_drop_unset(optional_keys))
    click.secho(str(kwargs_mask_label))
    labels = nima.segment(im, **kwargs_mask_label)

    # Measure
    kwargs_meas_props: dict[str, Any] = {"channels": channels}
    kwargs_meas_props["ratios_from_image"] = image_ratios
    if ratio_median_radii is not None:
        kwargs_meas_props["radii"] = ratio_median_radii
    click.secho(str(kwargs_meas_props))

    meas, _ = nima.measure(
        im,
        labels,
        channels_cl=channels_cl,
        channels_ph=channels_ph,
        **kwargs_meas_props,
    )

    r_cl_da = None
    r_ph_da = None
    if image_ratios:
        radii = kwargs_meas_props.get("radii", (7, 3))
        r_cl_da = nima.ratio(im, channels=channels_cl, radii=radii, mask=labels > 0)
        r_ph_da = nima.ratio(im, channels=channels_ph, radii=radii, mask=labels > 0)
    output_results(
        output,
        tiffstk,
        ff,
        meas,
        channels,
        bg_method,
        bgs,
        im,
        labels,
        r_cl=r_cl_da,
        r_ph=r_ph_da,
    )


def output_results(  # noqa: PLR0913
    output_dir: Path,
    tiffstk: Path,
    ff: dict[str, list[list[Figure]]],
    meas: dict[int, pd.DataFrame],
    channels: tuple[str, ...],
    bg_method: str,
    bgs: pd.DataFrame,
    im: xr.DataArray,
    labels: xr.DataArray,
    r_cl: xr.DataArray | None = None,
    r_ph: xr.DataArray | None = None,
) -> None:
    """Output results: csv tables and png images."""
    output_dir.mkdir(exist_ok=True)
    # Create file-named directory
    bname = output_dir / tiffstk.with_suffix("").name
    bname.mkdir(exist_ok=True)
    # Create PDF files
    for ch, llf in ff.items():
        pdf_file = bname / Path(f"bg-{ch}-{bg_method}.pdf")
        with PdfPages(pdf_file) as pp:
            for lf in llf:
                for f_i in lf:
                    pp.savefig(f_i)
    # Create CSV file
    column_order = ["C", "G", "R"]  # FIXME must be equal anyway in testing
    bgs[column_order].to_csv(bname / "bg.csv")
    # TODO: plt.close('all') or control mpl warning
    # Create measurement plots
    f = nima.plot_meas(bgs, meas, channels=channels)
    f.savefig(bname.with_name(bname.name + "_meas.png"))
    # Show all channels and labels
    fig = nima.plot_img(
        im,
        labels=labels,
        channels=sorted(channels),
        cmap=cm.inferno_r,
    )
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
    if r_cl is not None and r_ph is not None:
        objs = ndimage.find_objects(labels.to_numpy())
        for n, o in enumerate(objs):
            name = bname / Path(f"label{n + 1}_rcl.tif")
            tifffile.imwrite(
                name, r_cl.to_numpy()[o], compression="lzma", photometric="minisblack"
            )
            name = bname / Path(f"label{n + 1}_rpH.tif")
            tifffile.imwrite(
                name, r_ph.to_numpy()[o], compression="lzma", photometric="minisblack"
            )


# bima  ##################################################
@click.group()
@click.pass_context
@click.version_option()
@click.option("-o", "--output", type=PATH_OUT,
              help="Output path [default: *.tif, *.png].")  # fmt: skip
def bima(ctx: click.Context, output: Path) -> None:
    """Compute bias, dark and flat."""
    ctx.ensure_object(dict)
    ctx.obj["output"] = output


@bima.command()
@click.pass_context
@click.argument("fpath", type=PATH_IN)
def bias(ctx: click.Context, fpath: Path) -> None:
    """Compute the BIAS frame and estimate read noise.

    Parameters
    ----------
    ctx : click.Context
        Click context; ``ctx.obj["output"]`` holds the optional output path.
    fpath : Path
        Path to the bias stack (Light Off - 0 acquisition time).

    Notes
    -----
    Saves:
    1. BIAS image (.tif): Median projection.
    2. Plot (.png): Includes histograms, median projection, and visualization of
       hot pixels.
    3. Hot pixel coordinates and values (.csv): If hot pixels are detected.

    """
    if fpath.suffix == ".zip":
        with zipfile.ZipFile(fpath) as zf, zf.open(zf.namelist()[0]) as firstfile:
            store_val = tifffile.imread(BytesIO(firstfile.read()))
            # Convert to DataArray to be consistent with nima.hotpixels
            # Assume (T, Y, X) or (Y, X) from TiffFile
            if store_val.ndim == AXES_LENGTH_3D:
                store = xr.DataArray(store_val, dims=("T", "Y", "X"))
            else:
                store = xr.DataArray(store_val, dims=("Y", "X"))
    else:
        store = io.read_image(fpath)
        # store is TCZYX. We want to reduce over T.

    click.secho("Bias image-stack shape: " + str(store.shape), fg="green")

    # Compute median and std.
    # If store is from read_image (TCZYX), we reduce T.
    # If store is from zip (T, Y, X), we reduce T.
    if "T" in store.dims:
        bias_im = store.median(dim="T")
        err = store.std(dim="T")
    else:
        # Fallback for 2D or other dims without named T
        # (Unlikely for read_image, possible for zip path)
        bias_im = store
        err = xr.zeros_like(store)

    # Ensure 2D for hotpixels (squeeze C, Z if present)
    bias_im = bias_im.squeeze()
    err = err.squeeze()

    # hotpixels
    output = ctx.obj["output"] or fpath.with_name(f"{fpath.stem}_bias.png")
    _check_no_overwrite(
        [fpath], [output, output.with_suffix(".csv"), output.with_suffix(".tiff")]
    )

    err, hpix = _compute_bias_hpix(bias_im, err)
    if not hpix.empty:
        hpix.to_csv(output.with_suffix(".csv"), index=False)

    # percentiles on err (which is DataArray now).
    # .ravel() might need .values or flatten
    # err is likely 2D DataArray.
    p25, p50, p75 = np.percentile(err.to_numpy().ravel(), [25, 50, 75])
    err_str = sigfig.round(p50, p75 - p25)
    click.secho("Estimated read noise: " + err_str)
    tifffile.imwrite(
        output.with_suffix(".tiff"),
        bias_im,
        photometric="minisblack" if bias_im.ndim == AXES_LENGTH_3D else None,
    )
    # Output summary graphics.
    _plot_bias(bias_im, err, output, hpix, err_str)


@bima.command()
@click.pass_context
@click.option("--bias", "bias_fp", type=PATH_IN,
              help="File path to the bias stack (Light Off - Long acquisition time).")  # fmt: skip # noqa: E501
@click.option("--time", type=float,
              help="Acquisition time.")  # fmt: skip
@click.argument("fpath", type=PATH_IN)
def dark(ctx: click.Context, fpath: Path, bias_fp: Path | None, time: float) -> None:
    """Compute DARK.

    fpath : str
        Path to the dark stack (Light Off - Long acquisition time).

    Notes
    -----
    Saves:
    1. DARK image (.tif): Median projection.
    2. Plot (.png): Includes histograms, median projection, ...

    """
    dark_thr = 4.5
    output = ctx.obj["output"] or fpath.with_name(f"{fpath.stem}_dark.png")
    _check_no_overwrite(
        [fpath], [output, output.with_suffix(".png"), output.with_suffix(".tiff")]
    )
    store = io.read_image(fpath)
    click.secho("Dark image-stack shape: " + str(store.shape), fg="green")
    dark_im = store.median(dim="T") if "T" in store.dims else store
    dark_im = dark_im.squeeze()

    # Output summary graphics.
    title = os.fspath(output.with_suffix("").name)
    if bias_fp is not None:
        bias_im = io.read_image(bias_fp).squeeze()
        # Ensure alignment/broadcasting works
        dark_im -= bias_im
    if time:
        dark_im /= time
    tifffile.imwrite(
        output.with_suffix(".tiff"),
        dark_im.to_numpy(),
        photometric="minisblack" if dark_im.ndim == AXES_LENGTH_3D else None,
    )
    plt_img_profiles(dark_im, title, output)
    print(np.where(dark_im > dark_thr))


@bima.command()
@click.pass_context
@click.option("--bias", "bias_fp", type=PATH_IN,
              help="Path to the bias stack (Light Off - 0 acquisition time).")  # fmt: skip # noqa: E501
@click.argument("globpath", type=str)
def mflat(ctx: click.Context, globpath: str, bias_fp: Path | None) -> None:
    """Compute the flat field from a collection of (.tif) files.

    globpath : "glob expression"
        Glob pattern (enclosed in quotes) for a collection of (.tif) files.

    Notes
    -----
    Saves:
    1. FLAT image (.tif): Mean projection.
    2. Plot (.png): Includes histograms, mean projection, ...

    """
    image_sequence = tifffile.TiffSequence(globpath)
    stem = Path(Path(globpath).name.replace("*", "").replace("?", "")).stem
    output_path = ctx.obj["output"] or Path(f"{stem}_flat.tiff")
    _check_no_overwrite([Path(f) for f in image_sequence], _flat_outputs(output_path))
    sequence_info = f"{image_sequence.axes} {image_sequence.shape}"
    click.secho(sequence_info, fg="green")
    # Use synchronous scheduler to avoid distributed client issues in tests
    with dask.config.set(scheduler="synchronous"):
        # Stack TIFF files as a Dask array
        dask_array = da.stack(  # type: ignore[no-untyped-call]
            [
                da.from_array(tifffile.imread(file), chunks="auto")  # type: ignore[no-untyped-call]
                for file in image_sequence
            ],
            axis=0,
        )
        # Compute mean projection
        mean_projection = da.mean(dask_array, axis=0)
        # Compute the mean projection
        tprojection = mean_projection.compute()
    # Read the bias file (if provided)
    bias_frame = None
    if bias_fp:
        bias_frame = np.array(tifffile.imread(bias_fp))
    # Save the results
    _output_flat(output_path, tprojection, bias_frame)


@bima.command()
@click.pass_context
@click.option("--bias", "bias_fp", type=PATH_IN,
              help="Path to the bias stack (Light Off - 0 acquisition time).")  # fmt: skip # noqa: E501
@click.argument("fpath", type=PATH_IN)
def flat(ctx: click.Context, fpath: Path, bias_fp: Path | None) -> None:
    """Flat from (.tf8) file stack.

    fpath : str
        Path to the (.tf8) file containing the image data.

    Notes
    -----
    Saves:
    1. FLAT image (.tif): Mean projection.
    2. Plot (.png): Includes histograms, mean projection, ...

    """
    output = ctx.obj["output"] or fpath.with_name(f"{fpath.stem}_flat.tiff")
    _check_no_overwrite([fpath], _flat_outputs(output))
    stack = io.read_image(fpath)
    # store is TCZYX. We want mean over T.
    click.secho(f"Flat image-stack shape: {stack.shape}", fg="green")
    f = stack.mean(dim="T") if "T" in stack.dims else stack
    # Squeeze all singleton dimensions (Z, C, etc.)
    f = f.squeeze()
    with ProgressBar():  # type: ignore[no-untyped-call]
        tprojection = f.compute().to_numpy()
    bias_frame = None
    if bias_fp:
        bias_frame = np.array(tifffile.imread(bias_fp))
    _output_flat(output, tprojection, bias_frame)


def _flat_outputs(output: Path) -> list[Path]:
    """Return the paths written by :func:`_output_flat`."""
    return [output, output.with_stem(f"{output.stem}-raw"), output.with_suffix(".png")]


def _output_flat(
    output: Path, tprojection: ImFrame, bias_im: ImFrame | None = None
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
    tprojection : ImFrame
        2D array representing the raw flat field image (mean of frames).
    bias_im : ImFrame | None
        2D array representing the bias frame for subtraction.
        If None (default), no bias subtraction is performed.

    Notes
    -----
    The constant value (e.g., 20) added to 'tprojection' before subtracting
    'bias' in the function's implementation may need further review or
    adjustment based on the specific requirements of the flat field correction.

    """
    # Ensure the parent directories exist
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output.with_stem(f"{output.stem}-raw"), tprojection)
    if bias_im is None:
        flat_im = ndimage.gaussian_filter(tprojection, sigma=100)
    else:
        flat_im = ndimage.gaussian_filter(
            tprojection + 20 - bias_im, sigma=100
        )  # FIXME
        # MAYBE: consider skimage.filters.gaussian and  cmap=plt.cm.Set2_r
    flat_im /= flat_im.mean()
    tifffile.imwrite(output, flat_im)
    title = os.fspath(output.with_suffix("").name)
    plt_img_profiles(xr.DataArray(flat_im), title, output)


@bima.command()
@click.pass_context
@click.argument("fpath", type=PATH_IN)
def plot(ctx: click.Context, fpath: Path) -> None:
    """Plot profiles of a 2D image.

    fpath : str
        Path to the 2D image file (e.g. Bias or Dark).

    Notes
    -----
    A plot of profiles is saved as a '.png' file.

    """
    output = ctx.obj["output"] or fpath.with_suffix(".png")
    _check_no_overwrite([fpath], [output.with_suffix(".png")])
    img = io.read_image(fpath).squeeze()
    title = os.fspath(output.with_suffix("").name)
    plt_img_profiles(img, title, output)


def plt_img_profiles(
    img: xr.DataArray,
    title: str,
    output: Path,
    hpix: pd.DataFrame | None = None,
) -> None:
    """Compute and save image profiles graphics."""
    if img.ndim == AXES_LENGTH_2D:
        f = nima.plt_img_profile(img, title=title, hpix=hpix)
        f.savefig(output.with_suffix(".png"), dpi=250, facecolor="w")
        plt.close(f)
        # mark f = nima.plt_img_profile_2(img, title=title)
        # mark f.savefig(output.with_suffix(".2.png"), dpi=250, facecolor="w")
    else:
        for i in range(img.shape[0]):
            ch_title = f"{title} C:{i}"
            f = nima.plt_img_profile(img[i], title=ch_title)
            f.savefig(output.with_suffix(f".C{i}.png"), dpi=250, facecolor="w")
            plt.close(f)
            f = nima.plt_img_profile_2(img[i], title=ch_title)
            f.savefig(output.with_suffix(f".C{i}.2.png"), dpi=250, facecolor="w")
            plt.close(f)
