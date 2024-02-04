"""Tests for nima script."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import skimage.io
import skimage.measure
import tifffile as tff  # type: ignore
from click.testing import CliRunner, Result
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure

from nima.__main__ import bima, main

# tests path
tpath = Path(__file__).parent
# test data: (rootname, times)
rootnames = [("1b_c16_15", 4)]
ResultFolder = tuple[Path, tuple[str, int], Result]


@pytest.fixture(scope="module", params=rootnames)
def result_folder(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> ResultFolder:
    """Fixture for creating results folder and opening a sub-process."""
    tmpdir = tmp_path_factory.getbasetemp()
    filename = (tpath / "data" / request.param[0]).with_suffix(".tif")
    runner = CliRunner()
    result = runner.invoke(main, [str(filename), "G", "R", "C", "-o", str(tmpdir)])
    return tmpdir, request.param, result


def test_stdout(result_folder: ResultFolder) -> None:
    """It outputs the correct value for 'Times'."""
    out = result_folder[2].output
    assert result_folder[2].return_value is None
    assert result_folder[2].exit_code == 0
    assert (
        int(next(line for line in out.splitlines() if "Times:" in str(line)).split()[1])
        == result_folder[1][1]
    )


class TestOutputFiles:
    """It checks all output files."""

    @pytest.mark.parametrize("f", ["bg.csv", "label1.csv", "label2.csv", "label3.csv"])
    def test_csv(self: "TestOutputFiles", result_folder: ResultFolder, f: str) -> None:
        """It checks csv tables."""
        fp_expected = Path("tests/data/output/") / result_folder[1][0] / f
        # # TODO: why Path is needed?
        fp_result = result_folder[0] / result_folder[1][0] / f
        expected = pd.read_csv(fp_expected)
        result = pd.read_csv(fp_result)
        pd.testing.assert_frame_equal(expected, result, atol=1e-15)

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
    def test_tif(self, result_folder: ResultFolder, f: str) -> None:
        """It checks tif files: r_Cl, r_pH of segmented cells."""
        fp_expected = Path("tests/data/output/") / result_folder[1][0] / f
        fp_result = result_folder[0] / result_folder[1][0] / f
        expected = skimage.io.imread(fp_expected)  # type: ignore
        # FIXME: for utf8 encoding?
        result = skimage.io.imread(str(fp_result))  # type: ignore
        assert np.sum(result - expected) == pytest.approx(0, 2.3e-06)

    # @pytest.mark.mpl_image_compare(remove_text=True, tolerance=13)
    @pytest.mark.parametrize(("f", "tol"), [("_dim.png", 8.001), ("_meas.png", 20)])
    def test_png(self, result_folder: ResultFolder, f: str, tol: float) -> None:
        """It checks png files: saved segmentation and analysis."""
        fp_expected = tpath / "data" / "output" / "".join([result_folder[1][0], f])
        fp_result = result_folder[0] / "".join((result_folder[1][0], f))
        msg = compare_images(str(fp_expected), str(fp_result), tol)
        if msg:
            raise ImageComparisonFailure(msg)

    @pytest.mark.parametrize(
        "f", ["bg-C-li_adaptive.pdf", "bg-G-li_adaptive.pdf", "bg-R-li_adaptive.pdf"]
    )
    def test_pdf(self, result_folder: ResultFolder, f: str) -> None:
        """It checks pdf files: saved bg estimation."""
        fp_expected = Path("tests/data/output/") / result_folder[1][0] / f
        fp_result = result_folder[0] / result_folder[1][0] / f
        msg = compare_images(str(fp_expected), str(fp_result), 13)
        # Created by compare_images into tests/data folder
        rename = "_".join((fp_expected.name[:-4], "pdf.png"))
        fp_expected.with_name(rename).unlink()
        if msg:
            raise ImageComparisonFailure(msg)


def test_bias_mflat(tmp_path: Path) -> None:
    """Check `bias dflat` cli."""
    d = tmp_path
    tmpflt = d / "ff.tif"
    tmpraw = d / "ff-raw.tif"
    filename = str(Path("tests") / "data" / "test_flat*.tif")
    runner = CliRunner()
    result = runner.invoke(bima, ["-o", f"{tmpflt.resolve()}", "mflat", filename])
    assert str(3) in result.output
    test = tff.imread(tmpraw)
    expect = np.array(tff.imread(Path("tests") / "data" / "output" / "test_flat.tif"))
    np.testing.assert_allclose(np.array(test), expect)
    test = np.array(tff.imread(tmpflt))
    expect = np.array(
        tff.imread(Path("tests") / "data" / "output" / "test_flat_gaussnorm.tif")
    )
    np.testing.assert_allclose(test, expect)
    assert tmpflt.with_suffix(".png").exists()
