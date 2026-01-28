"""Input/Output operations for the NImA library.

This module replaces the legacy specific TiffReader with a standardized,
plugin-based approach using `bioio`. It supports reading a wide variety
of microscopy formats (TIFF, CZI, LIF, etc.) and returns data as
lazy-loaded xarray.DataArrays with standard dimensional metadata (TCZYX).
"""

from collections.abc import Sequence
from pathlib import Path

from bioio import BioImage
from xarray import DataArray


def read_image(fp: Path, channels: Sequence[str] | None = None) -> DataArray:
    """Read a microscopy image file using bioio.

    Reads image data into a lazy-loaded xarray.DataArray with standardized
    dimensions (TCZYX). Supports any format supported by the installed
    bioio plugins (e.g., OME-TIFF, CZI, LIF).

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

    Raises
    ------
    ValueError
        If the number of provided channels does not match the number of channels
        in the image file.

    Examples
    --------
    >>> from nima import io
    >>> img = io.read_image("tests/data/1b_c16_15.tif", channels=["G", "R", "C"])
    >>> img.coords["C"].values
    array(['G', 'R', 'C'], dtype='<U1')
    """
    img = BioImage(fp)
    # Get the dask-backed xarray directly
    # This property returns a lazy-loaded DataArray with standard dimensions
    data = img.xarray_dask_data

    # Preserve metadata in attrs
    # BioImage returns PhysicalPixelSizes(Z, Y, X)
    ps = img.physical_pixel_sizes
    if ps:
        data.attrs["physical_pixel_sizes"] = ps

    # Also good to keep original metadata for reference
    if img.metadata:
        data.attrs["metadata"] = img.metadata

    if channels:
        n_channels_data = data.sizes["C"]
        if len(channels) != n_channels_data:
            msg = (
                f"Channel mismatch: file has {n_channels_data}, "
                f"provided {len(channels)}"
            )
            raise ValueError(msg)

        # Assign new channel names to the 'C' coordinate
        data = data.assign_coords(C=list(channels))

    return data
