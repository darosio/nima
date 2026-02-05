"""Input/Output operations for the NImA library.

This module replaces the legacy specific TiffReader with a standardized,
plugin-based approach using `bioio`. It supports reading a wide variety
of microscopy formats (TIFF, CZI, LIF, etc.) and returns data as
lazy-loaded xarray.DataArrays with standard dimensional metadata (TCZYX).
"""

import atexit
import shutil
import tempfile
import warnings
from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from pathlib import Path
from pprint import pformat
from typing import Any

import numpy as np
from bioio import BioImage
from ome_types import OME
from xarray import DataArray


@dataclass()
class Channel:
    """Represent illumination-detection channel.

    Attributes
    ----------
    wavelength : int
        Illumination wavelength.
    attenuation : float
        Illumination attenuation.
    gain : float
        Detector gain.
    binning : str
        Detector binning.
    filters : list[str]
        List of filters.
    """

    wavelength: int
    attenuation: float
    gain: float
    binning: str
    filters: list[str]

    def __repr__(self) -> str:
        """Represent most relevant metadata."""
        return (
            f"Channel(λ={self.wavelength}, attenuation={self.attenuation}, "
            f"gain={self.gain}, binning={self.binning}, "
            f"filters hash={np.array([hash(f) for f in self.filters]).sum()})"
        )


@dataclass(eq=True, frozen=True)
class StagePosition:
    """Dataclass representing stage position.

    Attributes
    ----------
    x : float | None
        Position in the X dimension.
    y : float | None
        Position in the Y dimension.
    z : float | None
        Position in the Z dimension.
    """

    x: float | None
    y: float | None
    z: float | None

    def __hash__(self) -> int:
        """Generate a hash value for the object based on its attributes."""
        return hash((self.x, self.y, self.z))

    def __repr__(self) -> str:
        """Represent most relevant metadata."""
        return f"\t\tXYZ={pformat((self.x, self.y, self.z))}"


@dataclass(eq=True)
class VoxelSize:
    """Dataclass representing voxel size.

    Attributes
    ----------
    x : float | None
        Size in the X dimension.
    y : float | None
        Size in the Y dimension.
    z : float | None
        Size in the Z dimension.
    """

    x: float | None
    y: float | None
    z: float | None

    def __hash__(self) -> int:
        """Generate a hash value for the object based on its attributes."""
        return hash((self.x, self.y, self.z))


class MultiplePositionsError(Exception):
    """Exception raised when a series contains multiple stage positions."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


@dataclass
class Metadata:
    """Dataclass representing core metadata.

    Attributes
    ----------
    rdr : InitVar[OME]
        OME object used to initialize the class.
    size_s : int
        Number of series (size in the S dimension).
    size_x : list[int]
        List of sizes in the X dimension.
    size_y : list[int]
        List of sizes in the Y dimension.
    size_z : list[int]
        List of sizes in the Z dimension.
    size_c : list[int]
        List of sizes in the C dimension.
    size_t : list[int]
        List of sizes in the T dimension.
    dimension_order : list[str]
        List of dimension order for each pixels.
    bits : list[int]
        List of bits per pixel.
    objective : list[str]
        List of objectives.
    name : list[str]
        List of series names.
    date : list[str]
        List of acquisition dates.
    stage_position : list[dict[StagePosition, tuple[int, int, int]]]
        List of {StagePosition: (T,C,Z)} for each `S`.
    voxel_size : list[VoxelSize]
        List of voxel sizes.
    channels : list[list[Channel]]
        Channels settings.
    tcz_deltat : list[list[tuple[int, int, int, float]]]
        Delta time for each T C Z.
    """

    ome: InitVar[OME]
    _ome: OME = field(init=False, repr=False)
    size_s: int = 1
    size_x: list[int] = field(default_factory=list)
    size_y: list[int] = field(default_factory=list)
    size_z: list[int] = field(default_factory=list)
    size_c: list[int] = field(default_factory=list)
    size_t: list[int] = field(default_factory=list)
    dimension_order: list[str] = field(default_factory=list)
    bits: list[int] = field(default_factory=list)
    objective: list[str | None] = field(default_factory=list)
    name: list[str] = field(default_factory=list)
    date: list[str | None] = field(default_factory=list)
    stage_position: list[dict[StagePosition, tuple[int, int, int]]] = field(
        default_factory=list
    )
    voxel_size: list[VoxelSize] = field(default_factory=list)
    channels: list[list[Channel]] = field(default_factory=list)
    tcz_deltat: list[list[tuple[int, int, int, float]]] = field(default_factory=list)

    def __repr__(self) -> str:
        """Represent most relevant metadata."""
        return (
            f"Metadata(S={self.size_s}, T={self.size_t}, C={self.size_c}, "
            f"Z={self.size_z}, Y={self.size_y}, X={self.size_x}, "
            f"order={self.dimension_order}\n"
            f"         Bits={self.bits}, Obj={self.objective}\n"
            f"         voxel size={pformat(self.voxel_size)}\n"
            f"         stage=\n{pformat(self.stage_position)}\n"
            f"         channels=\n{pformat(self.channels)})"
        )

    def __post_init__(self, ome: OME) -> None:
        """Consolidate all core metadata."""
        self.size_s = len(ome.images)
        for image in ome.images:
            pixels = image.pixels
            self.size_x.append(pixels.size_x)
            self.size_y.append(pixels.size_y)
            self.size_z.append(pixels.size_z)
            self.size_c.append(pixels.size_c)
            self.size_t.append(pixels.size_t)
            self.dimension_order.append(str(pixels.dimension_order.value))
            self.bits.append(pixels.significant_bits or 0)
            self.name.append(image.id)
            self.objective.append(
                image.objective_settings.id if image.objective_settings else None
            )
            self.date.append(
                str(image.acquisition_date) if image.acquisition_date else None
            )
            self.stage_position.append(self._get_stage_position(pixels.planes))
            self.voxel_size.append(
                VoxelSize(
                    pixels.physical_size_x,
                    pixels.physical_size_y,
                    pixels.physical_size_z,
                )
            )
            self.channels.append(
                [
                    Channel(
                        int(channel.light_source_settings.wavelength or 0)
                        if channel.light_source_settings
                        else 0,
                        channel.light_source_settings.attenuation or 0.0
                        if channel.light_source_settings
                        else 0.0,
                        float(channel.detector_settings.gain or 0.0)
                        if channel.detector_settings
                        else 0.0,
                        str(channel.detector_settings.binning.value)
                        if channel.detector_settings
                        and channel.detector_settings.binning
                        else "1x1",
                        [
                            d.id.replace("Filter:", "")
                            for d in (
                                channel.light_path.excitation_filters
                                if channel.light_path
                                else []
                            )
                        ],
                    )
                    for channel in pixels.channels
                ]
            )

            self.tcz_deltat.append(
                [
                    (
                        plane.the_t,
                        plane.the_c,
                        plane.the_z,
                        plane.delta_t or 0.0,
                    )
                    for plane in pixels.planes
                ]
            )
        self._ome = ome
        for attribute in [
            "size_x",
            "size_y",
            "size_z",
            "size_c",
            "size_t",
            "dimension_order",
            "bits",
            "name",
            "objective",
            "date",
            "voxel_size",
        ]:
            if len(set(getattr(self, attribute))) == 1:
                setattr(self, attribute, list(set(getattr(self, attribute))))
        for channel in self.channels[1:]:
            if channel != self.channels[0]:
                break
            self.channels = [channel]

    def _get_stage_position(
        self, planes: list[Any]
    ) -> dict[StagePosition, tuple[int, int, int]]:
        """Retrieve the stage positions from the given pixels."""
        pos_dict: dict[StagePosition, tuple[int, int, int]] = {}
        for plane in planes:
            x, y, z = plane.position_x, plane.position_y, plane.position_z
            pos = StagePosition(
                float(x) if x is not None else None,
                float(y) if y is not None else None,
                float(z) if z is not None else None,
            )
            t, c, z_idx = plane.the_t, plane.the_c, plane.the_z
            pos_dict.update({pos: (int(t), int(c), int(z_idx))})
        return pos_dict


def _handle_tf8_workaround(fp: Path) -> Path:
    """Create a temporary symlink with .tif extension for .tf8 files.

    This ensures compatibility with bioio plugins that rely on file extensions.
    The temporary directory is cleaned up on process exit.
    """
    tmp_dir = tempfile.mkdtemp(prefix="nima_tf8_")
    atexit.register(shutil.rmtree, tmp_dir, ignore_errors=True)

    tmp_fp = Path(tmp_dir) / fp.with_suffix(".tif").name
    try:
        tmp_fp.symlink_to(fp.resolve())
    except OSError:
        shutil.copy(fp, tmp_fp)

    warnings.warn(
        f"Renaming .tf8 to .tif in {tmp_fp} for bioio compatibility. "
        "Temporary file will be removed on exit.",
        UserWarning,
        stacklevel=2,
    )
    return tmp_fp


def read_image(fp: Path, channels: Sequence[str] | None = None) -> DataArray:
    """Read a microscopy image file using bioio.

    Reads image data into a lazy-loaded xarray.DataArray with standardized
    dimensions (TCZYX). Supports any format supported by the installed
    bioio plugins (e.g., OME-TIFF, CZI, LIF).

    For .tf8 files (BigTIFF from photonics iMic), it renames them temporarily
    to .tif to ensure bioio compatibility if bioio_bioformats is not used.

    Parameters
    ----------
    fp : Path
        Path to the image file.
    channels : Sequence[str] | None
        Optional list of channel names. If provided, these names will overwrite
        the channel coordinates in the returned DataArray. The length must match
        the number of channels in the image.

    Returns
    -------
    DataArray
        A lazy-loaded xarray DataArray containing the image data.
        Dimensions are standardized to 'TCZYX' (Time, Channel, Z, Y, X).
        Channel names (if available in metadata) are preserved in the
        coordinates.

    Examples
    --------
    >>> from nima import io
    >>> img = io.read_image("tests/data/1b_c16_15.tif", channels=["G", "R", "C"])
    >>> img.coords["C"].values
    array(['G', 'R', 'C'], dtype='<U1')

    Raises
    ------
    ValueError
        If the number of provided channels does not match the number of channels
        in the image file.
    Exception
        If the file cannot be read by bioio.
    """
    try:
        img = BioImage(fp)
    except Exception:
        if fp.suffix == ".tf8":
            # Fallback for tf8: try renaming to .tif in a temp location
            # This is a workaround for iMic BigTIFF files
            tmp_fp = _handle_tf8_workaround(fp)
            img = BioImage(tmp_fp)
        else:
            raise
    # Get the dask-backed xarray directly
    # This property returns a lazy-loaded DataArray with standard dimensions
    data = img.xarray_dask_data

    # Parse and attach structured metadata
    if img.metadata and isinstance(img.metadata, OME):
        # Store the raw OME metadata
        data.attrs["ome_metadata"] = img.metadata
        # Parse and attach the structured Metadata object
        md = Metadata(img.metadata)
        data.attrs["metadata"] = md

    if channels:
        n_channels_data = data.sizes["C"]
        if len(channels) != n_channels_data:
            msg = (
                f"Channel mismatch: file has {n_channels_data}, "
                f"provided {len(channels)}"
            )
            raise ValueError(msg)

        # Validate C, G, R wavelength ordering if all three are present
        if "metadata" in data.attrs:
            md = data.attrs["metadata"]
            # md.channels is list[list[Channel]], one list per series.
            # We're reading the first series (scene 0) by default.
            if md and md.channels:
                # Get channels for the first series
                current_channels_meta = md.channels[0]

                if len(current_channels_meta) == len(channels):
                    # Map provided channel name to metadata wavelength
                    # The order matches: channels[i] renames current_channels_meta[i]
                    name_to_wave = {
                        name: ch_meta.wavelength
                        for name, ch_meta in zip(
                            channels, current_channels_meta, strict=False
                        )
                    }

                    # Check if C, G, R are all present in the channel names
                    if {"C", "G", "R"}.issubset(name_to_wave.keys()):
                        w_c = name_to_wave["C"]
                        w_g = name_to_wave["G"]
                        w_r = name_to_wave["R"]

                        # Validate wavelength ordering: lambda_C < lambda_G < lambda_R
                        if not (w_c < w_g < w_r):
                            msg = (
                                f"Channel wavelength validation failed: "
                                f"Expected λ_C < λ_G < λ_R. "
                                f"Got C={w_c}nm, G={w_g}nm, R={w_r}nm."
                            )
                            warnings.warn(msg, UserWarning, stacklevel=2)

        # Assign new channel names to the 'C' coordinate
        data = data.assign_coords(C=list(channels))

    return data
