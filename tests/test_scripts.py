"""
Tests for nimg script.
"""
import filecmp
import os
import subprocess

import numpy as np
import pytest
import skimage.io
import skimage.measure

# test data: (rootname, times)
rootnames = [('1b_c16_15', 4)]


@pytest.fixture(scope='module', params=rootnames)
def result_folder(tmpdir_factory, request):
    tmpdir = tmpdir_factory.mktemp('nniimmgg')
    filename = os.path.join('tests', 'data', request.param[0] + '.tif')
    cmd_line = ['nimg', filename, 'G', 'R', 'C', '-o', tmpdir]
    p = subprocess.Popen(
        cmd_line, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    return tmpdir, request.param, p


def test_printout(result_folder):
    """test both warning for divide by zero and time points number."""
    stdout, stderr = result_folder[2].communicate()
    assert 'true_divide' in str(stderr)
    assert int(str(stdout).split('Times:')[1].split('\\n')[0]
               .strip()) == result_folder[1][1]


class TestOutputFiles:
    @pytest.mark.parametrize(
        'f', ['bg.csv', 'label1.csv', 'label2.csv', 'label3.csv'])
    def test_csv(self, result_folder, f):
        fp_expected = os.path.join('tests/data/output/', result_folder[1][0],
                                   f)
        fp_result = result_folder[0].join(os.path.join(result_folder[1][0], f))
        assert filecmp.cmp(fp_result, fp_expected)

    @pytest.mark.parametrize('f', [
        'label1_rpH.tif', 'label1_rcl.tif', 'label2_rpH.tif', 'label2_rcl.tif',
        'label3_rpH.tif', 'label3_rcl.tif'
    ])
    def test_tif(self, result_folder, f):
        fp_expected = os.path.join('tests/data/output/', result_folder[1][0],
                                   f)
        fp_result = result_folder[0].join(os.path.join(result_folder[1][0], f))
        expected = skimage.io.imread(fp_expected)
        result = skimage.io.imread(str(fp_result))  # for utf8 encoding?
        assert np.sum(result - expected) == 0

    @pytest.mark.parametrize('f, ssim', [('_dim.png', 0.93),
                                         ('_meas.png', 0.64)])
    def test_png(self, result_folder, f, ssim):
        fp_expected = os.path.join('tests/data/output/',
                                   result_folder[1][0] + f)
        fp_result = result_folder[0].join(result_folder[1][0] + f)
        expected = skimage.io.imread(fp_expected)
        result = skimage.io.imread(str(fp_result))
        assert skimage.measure.compare_ssim(
            expected, result, multichannel=True) > ssim
