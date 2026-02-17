"""Input/Output operations for the NImA library.

This module replaces the legacy specific TiffReader with a standardized,
plugin-based approach using `nima_io`. It supports reading a wide variety
of microscopy formats (TIFF, CZI, LIF, etc.) and returns data as
lazy-loaded xarray.DataArrays with standard dimensional metadata (TCZYX).
"""

from collections.abc import Sequence
from pathlib import Path

import nima_io.read as nio
from nima_io.read import Metadata
from xarray import DataArray

__all__ = ["Metadata", "read_image"]


def read_image(
    fp: Path,
    channels: Sequence[str] | None = None,
    *,
    stitch_tiles: bool = False,
) -> DataArray:
    """Read a microscopy image file using nima_io.

    Reads image data into a lazy-loaded xarray.DataArray with standardized
    dimensions (TCZYX). Supports any format supported by the installed
    bioio/nima_io plugins (e.g., OME-TIFF, CZI, LIF).

    Parameters
    ----------
    fp : Path
        Path to the image file.
    channels : Sequence[str] | None
        Optional list of channel names. If provided, these names will overwrite
        the channel coordinates in the returned DataArray. The length must match
        the number of channels in the image.
    stitch_tiles : bool
        If True and the file contains multiple series representing tiles (based
        on stage positions), they will be stitched into a single large image.

    Returns
    -------
    DataArray
        A lazy-loaded xarray DataArray containing the image data.
        Dimensions are standardized to 'TCZYX' (Time, Channel, Z, Y, X).
        Channel names (if available in metadata) are preserved in the
        coordinates.
    """
    if stitch_tiles:
        return nio.stitch_scenes(str(fp), channels=channels)
    return nio.read_image(str(fp), channels=channels)
