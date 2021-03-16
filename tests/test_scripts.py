"""Tests for nimg script."""
import os
import subprocess

import numpy as np
import pandas as pd
import pytest
import skimage.io
import skimage.measure
from matplotlib.testing.compare import compare_images
from matplotlib.testing.exceptions import ImageComparisonFailure

# test data: (rootname, times)
rootnames = [("1b_c16_15", 4)]


@pytest.fixture(scope="module", params=rootnames)
def result_folder(tmpdir_factory, request):
    """Fixture for creating results folder and opening a sub-process."""
    tmpdir = tmpdir_factory.mktemp("nniimmgg")
    filename = os.path.join("tests", "data", request.param[0] + ".tif")
    cmd_line = ["nimg", filename, "G", "R", "C", "-o", tmpdir]
    p = subprocess.Popen(cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tmpdir, request.param, p


def test_printout(result_folder):
    """It outputs the correct value for 'Times'."""
    stdout, stderr = result_folder[2].communicate()
    assert (
        int(str(stdout).split("Times:")[1].split("\\n")[0].strip())
        == result_folder[1][1]
    )


class TestOutputFiles:
    """It checks all output files."""

    @pytest.mark.parametrize("f", ["bg.csv", "label1.csv", "label2.csv", "label3.csv"])
    def test_csv(self, result_folder, f):
        """It checks csv tables."""
        fp_expected = os.path.join("tests/data/output/", result_folder[1][0], f)
        fp_result = result_folder[0].join(os.path.join(result_folder[1][0], f))
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
    def test_tif(self, result_folder, f):
        """It checks tif files: r_Cl, r_pH of segmented cells."""
        fp_expected = os.path.join("tests/data/output/", result_folder[1][0], f)
        fp_result = result_folder[0].join(os.path.join(result_folder[1][0], f))
        expected = skimage.io.imread(fp_expected)
        result = skimage.io.imread(str(fp_result))  # for utf8 encoding?
        assert np.sum(result - expected) == pytest.approx(0, 2.3e-06)

    @pytest.mark.parametrize("f, tol", [("_dim.png", 0.001), ("_meas.png", 0.1)])
    def test_png(self, result_folder, f, tol):
        """It checks png files: saved segmentation and analysis."""
        fp_expected = os.path.join("tests/data/output/", result_folder[1][0] + f)
        fp_result = result_folder[0].join(result_folder[1][0] + f)
        msg = compare_images(fp_expected, fp_result, tol)
        if msg:
            raise ImageComparisonFailure(msg)

    @pytest.mark.parametrize(
        "f", ["bg-C-li_adaptive.pdf", "bg-G-li_adaptive.pdf", "bg-R-li_adaptive.pdf"]
    )
    def test_pdf(self, result_folder, f):
        """It checks pdf files: saved bg estimation."""
        fp_expected = os.path.join("tests/data/output/", result_folder[1][0], f)
        fp_result = result_folder[0].join(os.path.join(result_folder[1][0], f))
        msg = compare_images(fp_expected, fp_result, 0.0001)
        os.remove(fp_expected[:-4] + "_pdf.png")
        if msg:
            raise ImageComparisonFailure(msg)
