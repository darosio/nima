"""Tests for nima script."""
from pathlib import Path
from typing import Any
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import skimage.io  # type: ignore
import skimage.measure  # type: ignore
from click.testing import CliRunner
from matplotlib.testing.compare import compare_images  # type: ignore
from matplotlib.testing.exceptions import ImageComparisonFailure  # type: ignore

from nima import __main__


# test data: (rootname, times)
rootnames = [("1b_c16_15", 4)]


@pytest.fixture(scope="module", params=rootnames)
def result_folder(tmpdir_factory: Any, request: Any) -> Tuple[Path, Any, Any]:
    # ) -> Tuple[Path, Any, subprocess.Popen[bytes]]: # requires python>=3.9
    """Fixture for creating results folder and opening a sub-process."""
    tmpdir = tmpdir_factory.mktemp("nniimmgg")
    filename = Path("tests/data") / "".join((request.param[0], ".tif"))
    runner = CliRunner()
    result = runner.invoke(
        __main__.main, [f"{filename.resolve()}", "G", "R", "C", "-o", tmpdir]
    )
    return tmpdir, request.param, result


def test_stdout(result_folder: Any) -> None:
    """It outputs the correct value for 'Times'."""
    out = result_folder[2].output
    assert result_folder[2].return_value is None
    assert result_folder[2].exit_code == 0
    assert (
        int([line for line in out.splitlines() if "Times:" in str(line)][0].split()[1])
        == result_folder[1][1]
    )


class TestOutputFiles:
    """It checks all output files."""

    @pytest.mark.parametrize("f", ["bg.csv", "label1.csv", "label2.csv", "label3.csv"])
    def test_csv(self, result_folder: str, f: str) -> None:
        """It checks csv tables."""
        fp_expected = Path("tests/data/output/") / result_folder[1][0] / f
        fp_result = Path(result_folder[0]) / result_folder[1][0] / f
        expected = pd.read_csv(fp_expected)
        result = pd.read_csv(fp_result)
        pd.testing.assert_frame_equal(expected, result, atol=1e-15)  # type: ignore

    @pytest.mark.parametrize(
        "f",
        [
            "label1_rpH.tif",
            "label1_rcl.tif",
            "label2_rpH.tif",
            "label2_rcl.tif",
            "label3_rpH.tif",
            "label3_rcl.tif",
        ],
    )
    def test_tif(self, result_folder: Any, f: str) -> None:
        """It checks tif files: r_Cl, r_pH of segmented cells."""
        fp_expected = Path("tests/data/output/") / result_folder[1][0] / f
        fp_result = Path(result_folder[0]) / result_folder[1][0] / f
        expected = skimage.io.imread(fp_expected)
        result = skimage.io.imread(str(fp_result))  # for utf8 encoding?
        assert np.sum(result - expected) == pytest.approx(0, 2.3e-06)

    # @pytest.mark.mpl_image_compare(remove_text=True, tolerance=13)
    @pytest.mark.parametrize(("f", "tol"), [("_dim.png", 8.001), ("_meas.png", 20)])
    def test_png(self, result_folder: Any, f: str, tol: float) -> None:
        """It checks png files: saved segmentation and analysis."""
        fp_expected = Path("tests/data/output/") / "".join((result_folder[1][0], f))
        fp_result = Path(result_folder[0]) / "".join((result_folder[1][0], f))
        msg = compare_images(fp_expected, fp_result, tol)
        if msg:
            raise ImageComparisonFailure(msg)

    @pytest.mark.parametrize(
        "f", ["bg-C-li_adaptive.pdf", "bg-G-li_adaptive.pdf", "bg-R-li_adaptive.pdf"]
    )
    def test_pdf(self, result_folder: Any, f: str) -> None:
        """It checks pdf files: saved bg estimation."""
        fp_expected = Path("tests/data/output/") / result_folder[1][0] / f
        fp_result = Path(result_folder[0]) / result_folder[1][0] / f
        msg = compare_images(fp_expected, fp_result, 13)
        # Created by compare_images into tests/data folder
        stem = "_".join((fp_expected.stem, "pdf"))
        fp_expected.with_stem(stem).with_suffix(".png").unlink()
        if msg:
            raise ImageComparisonFailure(msg)
